\# Dataset Card: IMDB Movie Reviews Sentiment Analysis



\*\*Dataset Name:\*\* IMDB Large Movie Review Dataset

\*\*Version:\*\* 1.0 (with custom preprocessing)

\*\*Date:\*\* December 14, 2025

\*\*License:\*\* Research and Educational Use



---



\## Dataset Summary



The IMDB Movie Review Dataset contains 50,000 highly polar movie reviews from the Internet Movie Database. This dataset was created for binary sentiment classification research and has become a standard benchmark in natural language processing.



\*\*Key Statistics:\*\*

\- \*\*Total Reviews:\*\* 50,000

\- \*\*Training Set:\*\* 22,500 (after validation split)

\- \*\*Validation Set:\*\* 2,500

\- \*\*Test Set:\*\* 25,000

\- \*\*Classes:\*\* 2 (Positive, Negative)

\- \*\*Balance:\*\* Perfect 50/50 split

\- \*\*Language:\*\* English

\- \*\*Domain:\*\* Movie reviews

\- \*\*Collection Period:\*\* Pre-2011



---



\## Data Fields



| Field | Type | Description | Example |

|-------|------|-------------|---------|

| `text` | string | Movie review content (HTML removed) | "This movie was fantastic..." |

| `label` | integer | Sentiment (0=negative, 1=positive) | 1 |

| `split` | string | Data partition (train/val/test) | "train" |



\###



 Label Distribution



```

Negative (label=0): 25,000 reviews (50.0%)

Positive (label=1): 25,000 reviews (50.0%)



No class imbalance - perfectly balanced dataset

```



\### Text Statistics



| Metric | Value |

|--------|-------|

| \*\*Mean Length\*\* | 234 words / 1,325 characters |

| \*\*Median Length\*\* | 176 words / 987 characters |

| \*\*Shortest Review\*\* | 10 words / 52 characters |

| \*\*Longest Review\*\* | 2,470 words / 13,704 characters |

| \*\*Vocabulary Size\*\* | ~43,000 unique words |



---



\## Data Collection



\### Source

\- \*\*Origin:\*\* Internet Movie Database (IMDB) website

\- \*\*URL:\*\* https://ai.stanford.edu/~amaas/data/sentiment/

\- \*\*Collectors:\*\* Stanford AI Lab (Maas et al., 2011)

\- \*\*Method:\*\* Web scraping of publicly available reviews



\### Collection Methodology



\*\*Labeling Strategy:\*\*

\- Reviews with \*\*rating ≤ 4\*\* (out of 10) → Negative

\- Reviews with \*\*rating ≥ 7\*\* (out of 10) → Positive

\- Reviews with rating 5-6 → Excluded (ambiguous sentiment)



\*\*This creates:\*\*

\- Strong sentiment signals (no neutral/mixed reviews)

\- Clear binary classification task

\- But misses nuanced opinions



\### Time Period

\- \*\*Collection:\*\* Before 2011

\- \*\*Review Dates:\*\* Varies (some reviews of older movies)

\- \*\*Language Evolution:\*\* May not capture modern slang/terminology



---



\## Preprocessing Applied



\### Original Data Issues

\- HTML tags (`<br />`, `<p>`, `</div>`)

\- URLs and hyperlinks

\- Inconsistent whitespace and encoding

\- Special characters



\### Preprocessing Steps



\*\*1. HTML Removal\*\*

```python

text = re.sub(r'<.\\\\\\\*?>', '', text)

```

Removes: `<br />`, `<p>`, `<div>`, etc.



\*\*2. URL Removal\*\*

```python

text = re.sub(r'http\\\\\\\\S+|www.\\\\\\\\S+', '', text)

```

Removes: Links to IMDB pages, external sites



\*\*3. Whitespace Normalization\*\*

```python

text = ' '.join(text.split())

```

Fixes: Multiple spaces, tabs, newlines



\*\*4. Encoding\*\*

\- UTF-8 encoding enforced

\- Special characters preserved (é, ñ, etc.)



\### What Was PRESERVED



✅ \*\*Kept:\*\*

\- Punctuation (!!!, ???, ...)

\- Capitalization (AMAZING, Great)

\- Spelling errors (realistic user content)

\- Contractions (don't, won't, it's)

\- Slang and informal language



❌ \*\*Not Removed:\*\*

\- Stop words (kept for baseline models)

\- Numbers and dates

\- Proper nouns (actor names, movie titles)



\### Data Validation



\*\*Quality Checks:\*\*

\- Schema validation with Pandera

\- No null values found

\- Label distribution verified (exactly 50/50)

\- Text length sanity checks

\- Duplicate detection (none found)



\*\*Checksum:\*\*

\- MD5: `7c2ac02c03563afcf9b574c7e56c153a`

\- Use for verification of downloads



---



\## Data Splits



\### Split Strategy



\*\*Original Split:\*\* 50/50 train/test by dataset creators



\*\*Modified Split (Our Version):\*\*

\- Created 10% validation set from training data

\- Stratified sampling (maintained 50/50 balance)

\- Random seed: 42 (reproducible)



\### Final Split Sizes



| Split | Total | Positive | Negative | Percentage |

|-------|-------|----------|----------|------------|

| \*\*Train\*\* | 22,500 | 11,250 | 11,250 | 45% |

| \*\*Validation\*\* | 2,500 | 1,250 | 1,250 | 5% |

| \*\*Test\*\* | 25,000 | 12,500 | 12,500 | 50% |

| \*\*Total\*\* | 50,000 | 25,000 | 25,000 | 100% |



\*\*Why This Split:\*\*

\- Large test set (50%) for reliable evaluation

\- Small validation (5%) sufficient for hyperparameter tuning

\- Training set (45%) large enough for deep learning



---



\## Known Limitations



\### 1. Domain Specificity



\*\*Issue:\*\* Dataset contains only movie reviews



\*\*Implications:\*\*

\- May not generalize to product reviews

\- Different from restaurant/hotel reviews

\- Entertainment-specific language patterns

\- Cultural references to movies



\*\*Impact on Your Results:\*\*

\- Models may struggle on other domains

\- Out-of-distribution accuracy: 68-90% (from stress tests)



\### 2. Temporal Bias



\*\*Issue:\*\* Data collected before 2011



\*\*Implications:\*\*

\- Missing modern slang ("lit", "slay", "fire")

\- No COVID-era language

\- Dated cultural references

\- Language evolution not captured



\*\*Recommendations:\*\*

\- Test on recent reviews before deployment

\- Monitor for drift in production

\- Consider periodic retraining



\### 3. Binary Classification Artifact



\*\*Issue:\*\* Only extremely positive/negative reviews



\*\*Implications:\*\*

\- No neutral class (ratings 5-6 excluded)

\- Artificially polarized dataset

\- Missing mixed sentiment ("good acting, bad plot")

\- Real-world has more nuance



\*\*Your Stress Test Results:\*\*

\- Neutral-negative cases: 0-100% accuracy (inconsistent)

\- Models confused by mixed sentiment



\### 4. Language and Demographics



\*\*Issue:\*\* English-only, unknown demographics



\*\*Limitations:\*\*

\- Not multilingual

\- No demographic data (age, gender, location)

\- Likely biased toward US/UK users

\- May underrepresent minority perspectives



\*\*Your Bias Analysis Results:\*\*

\- Gender mentions: 999/1000 reviews (natural for movies)

\- Race mentions: 88/1000 reviews (8.8%)

\- Age mentions: 353/1000 reviews (35.3%)

\- \*\*No significant prediction bias detected\*\* ✅



\### 5. Platform Bias



\*\*Issue:\*\* IMDB user population



\*\*Implications:\*\*

\- Movie enthusiasts overrepresented

\- Casual viewers underrepresented

\- Self-selection bias

\- May not reflect general population



---



\## Known Biases



\### 1. Selection Bias



\*\*Description:\*\* Only IMDB users who chose to write reviews



\*\*Effect:\*\*

\- More passionate opinions (very good or very bad)

\- Underrepresents neutral/indifferent viewers

\- Skews toward movie buffs



\*\*Mitigation:\*\* Understand target audience differs from dataset



\### 2. Rating Threshold Bias



\*\*Description:\*\* Ratings 5-6 excluded creates artificial polarization



\*\*Effect:\*\*

\- No "it was okay" reviews

\- Forces binary decision

\- Loses nuance



\*\*Your Results:\*\*

\- Models struggle with truly neutral statements

\- "Not bad, not great" → unpredictable (stress tests)



\### 3. Content Bias



\*\*Description:\*\* Popular movies overrepresented



\*\*Observations:\*\*

\- Blockbusters have more reviews

\- Independent films underrepresented

\- English-language films favored

\- Certain genres may dominate



\### 4. Temporal Bias



\*\*Description:\*\* Older movies reviewed with different standards



\*\*Examples:\*\*

\- Special effects judged by era

\- Cultural context changes

\- Nostalgia factor



\### 5. Demographic Bias (Analyzed)



\*\*Your Safety Evaluation Results:\*\*



\*\*Gender (FALSE POSITIVE ALERT):\*\*

```

Reviews mentioning gender: 999/1000

Positive rate: 49.1%

Other positive rate: 0% (only 1 sample)

Difference: 49.1%



INTERPRETATION: Not actual bias - gender words (he/she/actor/actress) 

appear naturally in movie reviews. This is expected and acceptable.

```



\*\*Race:\*\*

```

Reviews mentioning race: 88/1000 (8.8%)

Positive rate: 48.9%

Other positive rate: 49.1%

Difference: 0.3%



VERDICT: No significant bias ✅

```



\*\*Age:\*\*

```

Reviews mentioning age: 353/1000 (35.3%)

Positive rate: 54.1%

Other positive rate: 46.4%

Difference: 7.7%



VERDICT: Minimal bias, within acceptable range (<10%) ✅

```



---



\## Ethical Considerations



\### Intended Uses ✅



\*\*Approved Applications:\*\*

\- Academic research on sentiment analysis

\- Machine learning education and training

\- Algorithm benchmarking

\- Prototype development

\- Technical blog posts and tutorials

\- Non-commercial research projects



\### Prohibited Uses ❌



\*\*Not Approved For:\*\*

\- Surveillance of individuals

\- Inferring personal attributes of reviewers

\- Making decisions about individuals

\- Discriminatory filtering

\- Automated content censorship without oversight

\- Commercial use without proper licensing

\- Manipulating public opinion

\- Identifying anonymous users



\### Privacy Considerations



\*\*Data Status:\*\*

\- ✅ Reviews are publicly available on IMDB

\- ✅ No personally identifiable information (PII)

\- ✅ Usernames not included in dataset

\- ✅ Cannot link reviews to individuals

\- ✅ Aggregated for research purposes



\*\*Recommendations:\*\*

\- Do not attempt to re-identify reviewers

\- Do not combine with other datasets for identification

\- Respect original public/private status



\### Sensitive Content



\*\*Your Safety Evaluation Found:\*\*

\- 128/1000 reviews flagged for "toxic" keywords

\- \*\*ALL were false positives\*\* - describing movie plots

\- Examples: "kill" (action movies), "violence" (describing scenes)

\- 18/1000 mentioned sensitive topics (suicide, terrorism in plots)



\*\*Assessment:\*\*

\- ✅ No hate speech detected

\- ✅ No discriminatory content

\- ✅ Content describes fiction, not real events

\- ✅ Safe for research and development



\### Fairness and Bias Mitigation



\*\*What We Did:\*\*

1\. Analyzed demographic keyword predictions

2\. Measured sentiment differences across groups

3\. Documented all findings transparently

4\. Provided mitigation recommendations



\*\*What You Should Do:\*\*

1\. Monitor predictions on protected groups

2\. Regular bias audits (quarterly)

3\. Track accuracy by demographic mentions

4\. Implement confidence thresholds for edge cases

5\. Allow user feedback and appeals



---



\## Dataset Quality Assessment



\### Strengths ✅



1\. \*\*Large Scale\*\*

   - 50,000 samples sufficient for deep learning

   - Enables robust evaluation (25,000 test samples)



2\. \*\*Balanced Classes\*\*

   - Perfect 50/50 split

   - No class imbalance issues

   - Fair comparison of precision/recall



3\. \*\*Clean Labels\*\*

   - Based on actual ratings (not subjective annotation)

   - High inter-annotator agreement (implicit)

   - Clear sentiment signals



4\. \*\*Diverse Vocabulary\*\*

   - ~43,000 unique words

   - Rich expressions of sentiment

   - Varied writing styles



5\. \*\*Real-World Language\*\*

   - Authentic user content

   - Natural language patterns

   - Includes typos, slang, informal style



6\. \*\*Well-Studied Benchmark\*\*

   - Hundreds of research papers

   - Standard comparison point

   - Reproducible results



\### Weaknesses ❌



1\. \*\*Domain-Specific\*\*

   - Movies only, doesn't generalize

   - Entertainment context differs from products



2\. \*\*Binary Labels\*\*

   - No neutral class

   - Missing mixed sentiment

   - Artificially polarized



3\. \*\*Temporal Limitations\*\*

   - Pre-2011 data

   - Missing modern language

   - Dated references



4\. \*\*English Only\*\*

   - Not multilingual

   - Limits global applicability



5\. \*\*Platform Bias\*\*

   - IMDB users != general population

   - Self-selection effects



6\. \*\*No Metadata\*\*

   - No demographic information

   - No movie genres

   - No review dates

   - No user information



---



\## Maintenance and Updates



\### Current Version

\- \*\*Version:\*\* 1.0 (custom preprocessing applied)

\- \*\*Processing Date:\*\* December 14, 2025

\- \*\*Base Dataset:\*\* Original IMDB v1.0 (unchanged)



\### Versioning

\- Base dataset is static (no updates planned)

\- Our preprocessing is versioned

\- Changes documented in this card



\### Future Considerations



\*\*Potential Improvements:\*\*

1\. Add more recent reviews (post-2011)

2\. Include neutral class (ratings 5-6)

3\. Collect demographic metadata

4\. Add aspect-level annotations

5\. Multilingual expansion



\*\*Not Planned:\*\*

\- Original dataset maintainers not updating

\- Static benchmark is useful for comparison

\- Community can create derived versions



\### Contact Information



\*\*For This Dataset Card:\*\*

\- Maintainer: \[Your Name]

\- Email: \[Your Email]

\- Project: Week 4 NLP \& LLMs Internship



\*\*For Original Dataset:\*\*

\- Authors: Andrew Maas, et al.

\- Institution: Stanford University

\- Paper: ACL 2011

\- URL: https://ai.stanford.edu/~amaas/data/sentiment/



---



\## Usage Examples



\### Loading the Dataset



```python

from src.data.loader import load\\\\\\\_processed\\\\\\\_data



\\\\# Load specific split

train\\\\\\\_df = load\\\\\\\_processed\\\\\\\_data('train')

val\\\\\\\_df = load\\\\\\\_processed\\\\\\\_data('val')

test\\\\\\\_df = load\\\\\\\_processed\\\\\\\_data('test')



\\\\# Load all data

all\\\\\\\_df = load\\\\\\\_processed\\\\\\\_data()



print(f"Train: {len(train\\\\\\\_df)} samples")

print(f"Val: {len(val\\\\\\\_df)} samples")

print(f"Test: {len(test\\\\\\\_df)} samples")

```



\### Example Reviews



\*\*Positive Review:\*\*

```

Text: "Brilliant film! The acting was superb, the cinematography 

\\\&nbsp;      stunning, and the plot kept me engaged throughout. One of 

\\\&nbsp;      the best movies I've seen this year. Highly recommended!"

Label: 1 (Positive)

Features: "brilliant", "superb", "stunning", "best", "recommended"

```



\*\*Negative Review:\*\*

```

Text: "Terrible waste of time. The plot made no sense, the acting 

\\\&nbsp;      was wooden, and I was bored throughout. Save your money and 

\\\&nbsp;      skip this one."

Label: 0 (Negative)

Features: "terrible", "waste", "no sense", "wooden", "bored"

```



\*\*Challenging Case (Negation):\*\*

```

Text: "Not bad at all, actually quite enjoyable."

Label: 1 (Positive)

Challenge: Contains "not" and "bad" (negative words)

Your Model Results: Baseline 0% accuracy, LoRA 33% accuracy on similar cases

```



---



\## References



\### Original Paper



```bibtex

@InProceedings{maas-EtAl:2011:ACL-HLT2011,

\\\&nbsp; author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  

\\\&nbsp;              Pham, Peter T.  and  Huang, Dan  and  

\\\&nbsp;              Ng, Andrew Y.  and  Potts, Christopher},

\\\&nbsp; title     = {Learning Word Vectors for Sentiment Analysis},

\\\&nbsp; booktitle = {Proceedings of the 49th Annual Meeting of the 

\\\&nbsp;              Association for Computational Linguistics: 

\\\&nbsp;              Human Language Technologies},

\\\&nbsp; month     = {June},

\\\&nbsp; year      = {2011},

\\\&nbsp; address   = {Portland, Oregon, USA},

\\\&nbsp; publisher = {Association for Computational Linguistics},

\\\&nbsp; pages     = {142--150},

\\\&nbsp; url       = {http://www.aclweb.org/anthology/P11-1015}

}

```



\### Related Work

\- Used in 500+ research papers (Google Scholar)

\- Standard NLP benchmark

\- Part of TensorFlow Datasets

\- Available in HuggingFace Datasets

\- Included in PyTorch torchtext



---



\## Changelog



\### Version 1.0 (December 14, 2025)

\- Applied custom preprocessing (HTML/URL removal)

\- Created 10% validation split

\- Generated stress test subsets (63 adversarial cases)

\- Added Pandera schema validation

\- Documented biases and limitations

\- Conducted safety evaluation

\- Measured actual model performance



\### Base Dataset (2011)

\- Original collection by Stanford AI Lab

\- 50,000 reviews from IMDB

\- Binary sentiment labels

\- Public release for research



---



\## Download and Verification



\### Download Commands

```bash

\\\\# Download original dataset

wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb\\\\\\\_v1.tar.gz



\\\\# Verify integrity

md5sum aclImdb\\\\\\\_v1.tar.gz

\\\\# Expected: 7c2ac02c03563afcf9b574c7e56c153a



\\\\# Extract

tar -xzf aclImdb\\\\\\\_v1.tar.gz



\\\\# Process with our scripts

python scripts/download\\\\\\\_data.py

python -m src.data.loader

```



\### File Structure

```

aclImdb/

├── train/

│   ├── pos/      # 12,500 positive reviews

│   ├── neg/      # 12,500 negative reviews

│   └── unsup/    # Unlabeled data (not used)

├── test/

│   ├── pos/      # 12,500 positive reviews

│   └── neg/      # 12,500 negative reviews

└── README

```



---



\## Summary



\*\*The IMDB Movie Review Dataset is:\*\*

\- ✅ Large-scale (50,000 samples)

\- ✅ Balanced (50/50 classes)

\- ✅ Clean labels (rating-based)

\- ✅ Well-studied benchmark

\- ✅ Real-world language

\- ⚠️ Domain-specific (movies only)

\- ⚠️ Temporal limitations (pre-2011)

\- ⚠️ Binary only (no neutral)

\- ✅ Safe for research use

\- ✅ No significant bias in predictions



\*\*Recommended for:\*\* Sentiment analysis research, ML education, algorithm benchmarking, prototype development.



\*\*Your Results:\*\* Models achieved 88-91% accuracy on this dataset, demonstrating its utility as a challenging but solvable benchmark.



---



\*Last Updated: December 14, 2025\*

\*Format: Based on Datasheets for Datasets (Gebru et al., 2018)\*

\*Project: Week 4 NLP \& LLMs Internship\*

