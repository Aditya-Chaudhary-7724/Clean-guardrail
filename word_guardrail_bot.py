import os
import sys
from pathlib import Path
from docx import Document
from transformers import pipeline
from audio_guardrail import analyze_audio_file
import time

class WordPoisonGuardrailBot:
    def __init__(self):
        print("🤖 Initializing Word Poison Guardrail Bot...")
        print("📦 Loading ModernBERT security model...")
        try:
            self.security_classifier = pipeline(
                "text-classification",
                model="gincioks/cerberus-modernbert-base-v1.0",
                device=-1
            )
            print("✅ ModernBERT security model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def get_security_score_for_segment(self, text: str) -> tuple:
        """Analyze a text segment and return (score, is_malicious, details)"""
        if not text.strip():
            return 0.0, False, "Empty text"

        result = self.security_classifier(text[:512])
        label, score = result[0]['label'], result[0]['score']

        if label.upper() == 'INJECTION':
            return score, True, f"🚨 Malicious content detected (confidence: {score:.1%})"
        else:
            return 1.0 - score, False, f"✅ Content appears safe (confidence: {score:.1%})"

    def analyze_document(self, filepath: str) -> dict:
        """Analyze a Word document and return comprehensive results"""
        try:
            # Validate file exists
            if not os.path.exists(filepath):
                return {"error": f"File not found: {filepath}"}

            # Validate file extension
            if not filepath.lower().endswith(('.docx', '.doc')):
                return {"error": "Only .docx and .doc files are supported"}

            # Extract text from document
            doc = Document(filepath)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

            if not paragraphs:
                return {"error": "Document contains no readable text"}

            full_text = "\n".join(paragraphs)

            # Analyze each paragraph
            results = []
            max_score = 0.0
            malicious_count = 0

            for idx, paragraph in enumerate(paragraphs, start=1):
                # Skip very short paragraphs
                if len(paragraph) < 20:
                    continue
                score, is_malicious, details = self.get_security_score_for_segment(paragraph)
                results.append({
                    "paragraph": idx,
                    "preview": paragraph[:100] + ("…" if len(paragraph) > 100 else ""),
                    "score": score,
                    "is_malicious": is_malicious,
                    "details": details
                })
                if score > max_score:
                    max_score = score
                if is_malicious:
                    malicious_count += 1

            # Overall verdict
            if max_score > 0.5:
                verdict = "🔴 BLOCKED - Poisoning Attack Detected"
                risk_level = "HIGH"
            elif max_score > 0.3:
                verdict = "🟡 SUSPICIOUS - Manual Review Recommended"
                risk_level = "MEDIUM"
            else:
                verdict = "🟢 SAFE - No Threats Detected"
                risk_level = "LOW"

            return {
                "filepath": filepath,
                "verdict": verdict,
                "risk_level": risk_level,
                "max_score": max_score,
                "total_paragraphs": len(paragraphs),
                "malicious_paragraphs": malicious_count,
                "details": results,
                "preview": full_text[:200] + ("…" if len(full_text) > 200 else "")
            }

        except Exception as e:
            return {"error": f"Error analyzing document: {e}"}

    def print_analysis_report(self, analysis: dict):
        """Print a formatted analysis report"""
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return

        print("\n" + "="*60)
        print("🛡️  WORD POISON GUARDRAIL - ANALYSIS REPORT")
        print("="*60)
        print(f"📁 File: {Path(analysis['filepath']).name}")
        print(f"🎯 Verdict: {analysis['verdict']}")
        print(f"⚠️  Risk Level: {analysis['risk_level']}")
        print(f"🧮 Threat Score: {analysis['max_score']:.3f}")
        print(f"📄 Paragraphs Analyzed: {analysis['total_paragraphs']}")
        if analysis['malicious_paragraphs'] > 0:
            print(f"🚨 Malicious Segments: {analysis['malicious_paragraphs']}\n")
            for entry in analysis['details']:
                if entry['is_malicious']:
                    print(f"   Paragraph {entry['paragraph']}: {entry['details']}")
                    print(f"   Preview: {entry['preview']}\n")
        else:
            print("✅ No malicious content detected\n")

        print("📝 Document Preview:")
        print(f"   {analysis['preview']}")
        print("="*60)

    def get_file_path_input(self) -> str:
        """Get file path from user with support for drag & drop"""
        while True:
            print("\n" + "="*50)
            print("📂 Please provide a Word document to analyze:")
            print("   • Type the full file path")
            print("   • Drag & drop the file into this terminal")
            print("   • Type 'quit' to exit")
            print("="*50)
            user_input = input("🎯 File path: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                return None

            # PowerShell prepends "& " when dragging a file – remove it
            if user_input.startswith('&'):
                # e.g. "& 'C:\\path\\file.docx'"
                user_input = user_input.lstrip('& ').strip()

            # Remove surrounding single or double quotes
            if (user_input.startswith('"') and user_input.endswith('"')) or \
               (user_input.startswith("'") and user_input.endswith("'")):
                user_input = user_input[1:-1]

            file_path = os.path.abspath(user_input)
            if os.path.exists(file_path):
                return file_path

            print(f"❌ File not found: {file_path}")
    
    def get_audio_file_path_input(self) -> str:
            while True:
                print("\n" + "="*50)
                print("🎤 Please provide an audio file to analyze:")
                print("   • Type the full file path")
                print("   • Drag & drop the file into this terminal")
                print("   • Type 'back' to return to main menu")
                print("="*50)
                user_input = input("🎯 Audio file path: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'back':
                    return None
                if user_input.startswith('&'):
                    user_input = user_input.lstrip('& ').strip()
                if (user_input.startswith('"') and user_input.endswith('"')) or \
                (user_input.startswith("'") and user_input.endswith("'")):
                    user_input = user_input[1:-1]
                file_path = os.path.abspath(user_input)
                if os.path.exists(file_path):
                    return file_path
                print(f"❌ File not found: {file_path}")

    def run_interactive_mode(self):
        while True:
            print("\n" + "="*50)
            print("Choose input type:")
            print(" 1. Analyze Word document")
            print(" 2. Analyze Audio file")
            print(" 3. Quit")
            choice = input("Enter choice (1/2/3): ").strip()

            if choice == '1':
                filepath = self.get_file_path_input()
                if filepath is None:
                    continue
                print(f"\n🔍 Analyzing document: {Path(filepath).name}")
                print("⏳ Please wait...")
                time.sleep(0.5)
                analysis = self.analyze_document(filepath)
                self.print_analysis_report(analysis)

            elif choice == '2':
                audio_path = self.get_audio_file_path_input()
                if audio_path is None:
                    continue
                print(f"\n🎧 Analyzing audio file: {Path(audio_path).name}")
                print("⏳ Please wait...")
                time.sleep(0.5)
                analyze_audio_file(audio_path, self)

            elif choice == '3':
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid option. Please enter 1, 2 or 3.")

def main():
    """Main entry point"""
    bot = WordPoisonGuardrailBot()
    bot.run_interactive_mode()

if __name__ == "__main__":
    main()
