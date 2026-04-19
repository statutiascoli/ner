import argparse
import json
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import List, Literal
from outlines import Generator, from_transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Levenshtein import distance
import re
import random, numpy as np
from urllib.parse import unquote


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


EntityType = Literal["PER", "LOC", "FES"]


class Entity(BaseModel):
    type: EntityType
    text: str


class NEROutput(BaseModel):
    entities: List[Entity]


SENTENCE_SPLIT_RE = re.compile(r"(?<=\.)\s+(?=[A-ZÀ-ÖØ-Ý])")


def split_sentences_with_offsets(text):
    parts = []
    start = 0

    for m in SENTENCE_SPLIT_RE.finditer(text):
        end = m.start()
        sentence_text = text[start:end]
        if sentence_text.strip():
            parts.append({
                "text": sentence_text,
                "start": start,
                "end": end,
            })
        start = m.end()

    if start < len(text):
        sentence_text = text[start:]
        if sentence_text.strip():
            parts.append({
                "text": sentence_text,
                "start": start,
                "end": len(text),
            })

    return parts


def find_occurrences(text, value, ci=False):
    if not value:
        return []
    if ci:
        return [(m.start(), m.end()) for m in re.finditer(re.escape(value), text, re.IGNORECASE)]
    return [(m.start(), m.end()) for m in re.finditer(re.escape(value), text)]


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\w+', text, re.UNICODE)]


def normalize_for_distance(s):
    return re.sub(r"\s+", "", s)


def safe_unquote(x):
    if "%" in x:
        x = unquote(x)
    if r"\u" in x:
        x = x.encode().decode("unicode_escape")
    return x


def find_best_match(text, value, used, max_distance=2):
    value_stripped = value.strip()
    value_tokens = tokenize(value_stripped)
    n = len(value_tokens)
    text_tokens = tokenize(text)

    candidates = []

    for i in range(len(text_tokens) - n + 1):
        window = text_tokens[i:i + n]
        start = window[0][1]
        end = window[-1][2]

        if (start, end) in used:
            continue

        source_slice = text[start:end]
        dist = distance(
            normalize_for_distance(source_slice),
            normalize_for_distance(value_stripped),
        )

        if dist <= max_distance:
            candidates.append((dist, start, end))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    dist, start, end = candidates[0]
    return (start, end, dist)


def resolve_entities(text, entities):
    resolved = []
    used = set()

    for ent in entities:
        predicted = safe_unquote(ent["text"]).strip()
        found = None

        for start, end in find_occurrences(text, predicted):
            if (start, end) not in used:
                found = (start, end)
                break
        
        if found is None:
            for start, end in find_occurrences(text, predicted, ci=True):
                if (start, end) not in used:
                    found = (start, end)
                    print(f"CI MATCH: '{predicted}' -> '{text[start:end]}'")
                    break
        
        if found is None:
            match = find_best_match(text, predicted, used)
            if match is not None:
                start, end, dist = match
                found = (start, end)
                print(f"FUZZY MATCH: '{predicted}' -> '{text[start:end]}' (dist={dist})")

        if found is None:
            print(f"NOT FOUND ENTITY: {predicted}\nTEXT: {text}\n")
            continue

        used.add(found)
        start, end = found
        resolved.append({
            "type": ent["type"],
            "text": text[start:end],
            "start": start,
            "end": end,
        })

    return resolved

 
def get_ner_prompt(text):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert annotator for 15th-century communal statutes "
                "written in medieval Italian. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""Extract named entities from the text below.
 
Return a JSON object with a single key "entities". Each entity must have:
- "type": one of "PER", "LOC", "FES"
- "text": the exact span as it appears in the source text

### PER - Named persons
Tag as PER any span that names one specific human individual, including full names
and shorter personal names, with or without patronymic or place of origin:
"Muctio de Johanni de Bernardo", "Joanni da Theramo", "Andreiuctio".
Do not tag groups, bare roles/titles, or devotional figures as PER.

### LOC - Named places
Tag as LOC any span that names one specific place, whether it appears as a proper name alone
or with a place-denoting noun:
"Appognano", "castello de Appognano", "ecclesia de Sancta Maria majore", "palazzo del Popolo".
Do not tag generic or descriptive location phrases alone: "ciptà", "palazo", "comune".
Do not tag any reference to Ascoli.

### FES - Named religious festivities
Tag as FES any span that names one liturgical celebration:
"festa de sancto Emidio", "Penthecoste", "venardì sancto".
Do not tag generic temporal references as FES: "domenicha", "sabato", "kalende".

### Rules
- Span must be exactly as in the source text, preserving original spelling.
- The entity must be one single contiguous span in the source text.
- Do not tag leading prepositions or articles unless they are inseparable from the name.
- In lists, extract each item separately and exactly as written; do not expand omitted context.
- Do not tag anaphoric or referential expressions when they refer to an entity without naming it.
- If no entities are present, return: {{"entities": []}}

Text:
{text}
""",
        },
    ]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data
   

def main(input_path, output_path, model_path):
   
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    ).to("cuda:0")
    hf_model.eval()
    
    print(f"Loaded model: {model_path}")

    model = from_transformers(hf_model, tokenizer)
    generator = Generator(model, NEROutput)

    records = load_json(input_path)
    output_records = []
 
    for idx, record in enumerate(records):
        text = record.get("text_plain")
        record_id = record.get("id", f"record_{idx}")

        sentences = split_sentences_with_offsets(text)
        merged_entities = []

        for sent_idx, sent in enumerate(sentences):
            sent_text = sent["text"]
            sent_start = sent["start"]

            messages = get_ner_prompt(sent_text)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
 
            result = generator(
                prompt,
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )

            try:
                parsed = NEROutput.model_validate_json(result)
                raw_entities = parsed.model_dump()["entities"]
                resolved_entities = resolve_entities(sent_text, raw_entities)
                for ent in resolved_entities:
                    doc_start = sent_start + ent["start"]
                    doc_end = sent_start + ent["end"]

                    merged_entities.append({
                        "type": ent["type"],
                        "text": text[doc_start:doc_end],
                        "start": doc_start,
                        "end": doc_end,
                    })
            except:
                print(f"ERROR on record {record_id}, sentence {sent_text}")
 
        output_record = {
            **record,
            "entities": merged_entities,
            "meta": {
                "model": model_path.split("/")[-1],
                "date": datetime.now(timezone.utc).isoformat()
            },
        }
        output_records.append(output_record)
        print(f"processed record {record_id}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(output_records)} records to '{output_path}'")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-m", "--model", required=True, help="Model path")

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
    )