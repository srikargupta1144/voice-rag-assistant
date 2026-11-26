# Data sources

All sample documents in this repository were written for this project. They describe two fictional companies, Acme Mutual Insurance Company and Kova Bank, and contain no real customer data, no real product terms and no names of real people. Phone numbers use the reserved 555-01xx range and web addresses use the reserved `.example` domain. Any resemblance to a real insurer's or bank's terms is coincidental and only reflects common industry vocabulary.

| File in `data/docs` | Format | Content | Origin |
| --- | --- | --- | --- |
| `acme_home_insurance_policy.pdf` | PDF, 7 pages | Fictional homeowners policy: coverages A to D, exclusions, deductibles, payments and cancellation, discounts | Written for this project; rendered from `data/pdf_sources/acme_home_insurance_policy.md` by `scripts/make_sample_pdfs.py` |
| `kova_rewards_card_terms.pdf` | PDF, 4 pages | Fictional credit card pricing: rewards, APRs, fees, payments and grace period | Written for this project; rendered from `data/pdf_sources/kova_rewards_card_terms.md` |
| `acme_claims_guide.md` | Markdown | Fictional claims process: reporting, emergency repairs, documents, adjuster, payment, appraisal | Written for this project |
| `kova_bank_account_faq.md` | Markdown | Fictional bank account FAQ: opening, fees, overdraft, savings rate, wires, mobile deposit, lost cards, closing | Written for this project |
| `kova_privacy_notice.txt` | Plain text | Fictional privacy notice: data collected, sharing, choices, retention, rights | Written for this project |

The golden question set in `eval/questions.jsonl` (30 answerable and 5 unanswerable questions) was written against these documents, with each answerable question labelled by source file, page (for PDFs) and a key phrase from the passage that answers it.

## License

The sample documents and the question set are released under the same MIT license as the code, so they can be reused freely.

## Models (downloaded, not committed)

| Model | Source | License |
| --- | --- | --- |
| BAAI/bge-small-en-v1.5 (embeddings) | Hugging Face | MIT |
| cross-encoder/ms-marco-MiniLM-L-6-v2 (re-ranker) | Hugging Face | Apache 2.0 |
| Systran/faster-whisper-small (speech-to-text) | Hugging Face | MIT |
| rhasspy/piper-voices en_US-lessac-medium (text-to-speech) | Hugging Face | See the voice's MODEL_CARD in the piper-voices repository; check the training dataset's terms before any commercial use |
| llama3.2:3b (LLM, through Ollama) | Ollama library | Llama 3.2 Community License |
