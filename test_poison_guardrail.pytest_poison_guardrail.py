import sys
from docx import Document
from transformers import pipeline

# Load ModernBERT security model
security_classifier = pipeline(
    "text-classification",
    model="gincioks/cerberus-modernbert-base-v1.0",
    device=-1
)

def get_security_score_for_segment(text: str) -> float:
    result = security_classifier(text[:512])
    label, score = result[0]['label'], result[0]['score']
    if label == 'INJECTION':
        print(f"  🚨 Malicious segment detected (score: {score:.3f}): {text[:60]}…")
        return score
    return 1.0 - score

def get_max_document_score(doc_text: str) -> float:
    # Split into non-empty paragraphs
    paragraphs = [p.strip() for p in doc_text.split('\n') if p.strip()]
    scores = [get_security_score_for_segment(p) for p in paragraphs]
    return max(scores) if scores else 0.0

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_poison_guardrail.py path/to/file.docx")
        sys.exit(1)

    # Extract full text
    filepath = sys.argv[1]
    doc = Document(filepath)
    full_text = "\n".join(p.text for p in doc.paragraphs)

    # Score by segment
    max_score = get_max_document_score(full_text)
    print(f"\n🧮 Maximum document threat score: {max_score:.3f}")

    # Decision
    if max_score > 0.5:
        label, risk = "🔴 POISONING ATTACK (BLOCKED)", "HIGH"
    elif max_score > 0.3:
        label, risk = "🟡 SUSPICIOUS (CAUTION)", "MEDIUM"
    else:
        label, risk = "🟢 BENIGN (SAFE)", "LOW"

    # Report
    print("\n" + "="*50)
    print("🛡️  MODERNBERT POISON GUARDRAIL")
    print("="*50)
    print(f"📁 File: {filepath}")
    print(f"🎯 Prediction: {label}")
    print(f"⚠️  Risk Level: {risk}")
    print(f"🧮 Threat Score: {max_score:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
