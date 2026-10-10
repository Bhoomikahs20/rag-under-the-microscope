# Baseline RAG Pipeline Error Analysis

## 1. Executive Summary

This error analysis examines the **14 incorrect answers** produced by the **Baseline RAG Pipeline** on the 50-question linked HotpotQA pilot benchmark (`data/processed/rag_baseline/rag_results.json`). 

While the baseline RAG pipeline achieved a **72.0% accuracy rate (36/50 correct)**—a **+62.0% absolute improvement over closed-book FLAN-T5-Base (10.0%)**—the 28.0% error rate highlights fundamental limitations in single-vector retrieval and small-scale generative multi-hop reasoning.

> [!NOTE]
> All original benchmark results, scores, indices, and generated outputs in `data/processed/rag_baseline/rag_results.json` remain 100% unchanged.

---

## 2. Failure Mode Taxonomy & Distribution

Each of the 14 failures was evaluated against the 6 standardized failure categories:

| Failure Category | Count | Share (%) | Primary Root Cause |
|---|---|---|---|
| **Retrieval Failure** | **6** | **42.9%** | Dense vector search (`all-MiniLM-L6-v2`) failed to retrieve the 2nd-hop document into top-3 ($K=3$), leaving the LLM with incomplete context. |
| **Multi-Hop Reasoning Failure** | **5** | **35.7%** | Both gold documents were successfully retrieved into prompt, but FLAN-T5-Base failed to execute multi-hop entity association, temporal comparison, or boolean logic. |
| **Generation Failure** | **2** | **14.3%** | FLAN-T5 echoed prompt prefix framing text (`"The relevant information to answer..."`) and truncated prior to completing the answer string. |
| **Gold-Answer / Data-Quality Issue** | **1** | **7.1%** | Benchmark ground truth label is flawed (`"British"` for a question asking for a city name, where city is absent in corpus). |
| **Insufficient Context** | **0** | **0.0%** | (No failure caused by chunk truncation; 250-word chunks contained complete document content). |
| **Answer-Format Mismatch** | **0** | **0.0%** | (Exact/F1 match metrics accurately captured semantic equivalence; formatting issues were subsumed under generation prompt repetition). |

---

## 3. Comprehensive Per-Question Error Breakdown

---

### Question 1: Index 3 (`5a8780385542996e4f30880e`)
- **Question:** *What character in the television special "The Pound Puppies" did the voice actor of Fred Jones from "Scooby-Doo" voice as?*
- **Gold Answer:** `Howler`
- **RAG Generated Answer:** `Cooler`
- **Failure Category:** `Multi-Hop Reasoning Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved** (100% ground-truth context available)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Frank Welker | `487732` | 0.6594 | **Yes** | Franklin Wendell Welker (born March 12, 1946) is an American actor... best known for his role as Fred Jones from the "Scooby-Doo" franchise... |
| 2 | Pound Puppies (film) | `37328049` | 0.6047 | **Yes** | The Pound Puppies is an animated television special... Characters included the Fonzie-styled leader Cooler (voiced by Dan Gilvezan)... and the goofy inventor aptly named Howler (voiced by Frank Welker)... |
| 3 | Pooh's Heffalump Movie | `1499129` | 0.2804 | No | Pooh's Heffalump Movie is a 2005 American animated film... |

- **Root Cause Analysis:** Both required documents were retrieved. `Frank Welker` links Fred Jones $\rightarrow$ Frank Welker, and `Pound Puppies (film)` lists both Dan Gilvezan $\rightarrow$ Cooler and Frank Welker $\rightarrow$ Howler. However, FLAN-T5-Base misassociated the voice actor with the wrong character entity, selecting `Cooler` instead of `Howler`.

---

### Question 2: Index 5 (`5ae5fb975542996de7b71aa8`)
- **Question:** *Edmundo "Eddie" Rodriguez was born in what Mexican state?*
- **Gold Answer:** `Jalisco`
- **RAG Generated Answer:** `Guadalajara`
- **Failure Category:** `Retrieval Failure`
- **Relevant Gold Document Retrieval Status:** **1/2 Retrieved** (Missing Document: `Guadalajara`)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Edmundo Rodriguez | `21757108` | 0.8273 | **Yes** | Edmundo "Eddie" Rodriguez (born April 22, 1970 in Guadalajara, Mexico) is a retired Mexican soccer player... |
| 2 | Theodosia Okoh | `40343708` | 0.2656 | No | Theodosia Salome Okoh... Ghanaian stateswoman... |
| 3 | Dustin Kensrue | `2893773` | 0.2599 | No | Dustin Michael Kensrue... vocalist... |

- **Root Cause Analysis:** Dense retrieval fetched `Edmundo Rodriguez` (Doc 1) which states he was born in *Guadalajara, Mexico*. However, it failed to retrieve `Guadalajara` (Doc 2), which contains the crucial geographical fact that Guadalajara is located in the state of *Jalisco*. Lacking Doc 2, the LLM extracted the city name (`Guadalajara`) directly from Doc 1.

---

### Question 3: Index 7 (`5a7257ce5542990c210a410d`)
- **Question:** *This English comedian wrote the British sitcom "Absolutely Fabulous"*
- **Gold Answer:** `Jennifer Saunders`
- **RAG Generated Answer:** `John Sullivan`
- **Failure Category:** `Multi-Hop Reasoning Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved** (100% ground-truth context available)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Absolutely Fabulous (series 3) | `49546885` | 0.5646 | **Yes** | ...however, the following year, Jennifer Saunders decided to write two specials titled "The Last Shout"... |
| 2 | Jennifer Saunders | `89605` | 0.3806 | **Yes** | Jennifer Jane Saunders... is an English comedian, screenwriter, and actress... |
| 3 | Only Fools and Horses | `200842` | 0.3640 | No | Only Fools and Horses is a British television sitcom created and written by John Sullivan... |

- **Root Cause Analysis:** Both gold documents were present in the context. However, Rank 3 contained a distractor document (`Only Fools and Horses` created by `John Sullivan`). FLAN-T5-Base was distracted by the entity in Rank 3 and erroneously attributed the authorship of *Absolutely Fabulous* to `John Sullivan`.

---

### Question 4: Index 13 (`5adf48b25542995ec70e8fb5`)
- **Question:** *In what city did Amir Khan and Lamont Peterson fight for the welterweight titles in 2011?*
- **Gold Answer:** `British`
- **RAG Generated Answer:** `London`
- **Failure Category:** `Gold-Answer / Data-Quality Issue`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved**

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Lamont Peterson | `7558034` | 0.7361 | **Yes** | Previously he won the WBA (Super) and IBF light welterweight titles in 2011 by controversially defeating Amir Khan... |
| 2 | Amir Khan (boxer) | `1290366` | 0.6890 | **Yes** | Amir Iqbal Khan (born 8 December 1986) is a British professional boxer... |
| 3 | The Good Fight | `52152672` | 0.3089 | No | The Good Fight is an American legal and political drama... |

- **Root Cause Analysis:** Dual root cause: (1) The benchmark gold answer label in HotpotQA is `"British"`, which is a nationality, not a city. (2) Neither retrieved corpus document contains the actual city where the fight occurred (Washington, D.C.). The model generated `London` based on Amir Khan's nationality in Doc 2.

---

### Question 5: Index 14 (`5a89c2715542993b751ca990`)
- **Question:** *Which was released first, God's Angry Man or The World's Best Prom?*
- **Gold Answer:** `God's Angry Man`
- **RAG Generated Answer:** `The World's Best Prom`
- **Failure Category:** `Multi-Hop Reasoning Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved** (100% ground-truth context available)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | The World's Best Prom | `3751418` | 0.5733 | **Yes** | The World's Best Prom is a 2006 documentary film... |
| 2 | God's Angry Man | `858143` | 0.4263 | **Yes** | God's Angry Man is a 1981 documentary film... |
| 3 | 17 Carat | `52427290` | 0.2589 | No | 17 Carat is the debut extended play... |

- **Root Cause Analysis:** Both gold documents were retrieved into prompt with exact release years (1981 vs 2006). FLAN-T5-Base failed temporal integer comparison ($1981 < 2006$) and falsely asserted that the 2006 film was released first.

---

### Question 6: Index 17 (`5a7e264055429965cec5eaa4`)
- **Question:** *Fulda Cathedral is the burial place of the leading figure in what?*
- **Gold Answer:** `Anglo-Saxon mission`
- **RAG Generated Answer:** `The relevant information to answer the above question is: Fulda Cathedral (German: "Fuldaer Dom" , also "Sankt Salvator") is the former abbey church of Fulda Abbey and the`
- **Failure Category:** `Generation Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved**

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Fulda Cathedral | `28227842` | 0.7684 | **Yes** | Fulda Cathedral... is the burial place of Saint Boniface... |
| 2 | Saint Boniface | `28172` | 0.3175 | **Yes** | Saint Boniface... was a leading figure in the Anglo-Saxon mission to the Germanic parts of the Frankish Empire... |
| 3 | Fouquieria | `18952512` | 0.1836 | No | Fouquieria is a genus of 11 species of desert plants... |

- **Root Cause Analysis:** Both documents were retrieved. However, FLAN-T5-Base suffered a generation loop, parroting the prompt lead-in instruction and reaching the 50-token generation cap before generating the target answer string.

---

### Question 7: Index 21 (`5a71532e5542994082a3e7d6`)
- **Question:** *In what year did the actor who played Edward Snowden act in a movie with Heath Ledger and Julia Styles?*
- **Gold Answer:** `1999`
- **RAG Generated Answer:** `2016`
- **Failure Category:** `Retrieval Failure`
- **Relevant Gold Document Retrieval Status:** **1/2 Retrieved** (Missing Document: `10 Things I Hate About You`)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Joseph Gordon-Levitt | `1817766` | 0.5847 | **Yes** | Joseph Leonard Gordon-Levitt... played Edward Snowden in Oliver Stone's biographical drama Snowden (2016)... |
| 2 | Hugh Bonneville | `1802353` | 0.3218 | No | Hugh Richard Bonneville Williams... |
| 3 | Tom Hardy | `1792975` | 0.3015 | No | Tom Hardy... English actor... |

- **Root Cause Analysis:** Vector search retrieved `Joseph Gordon-Levitt` (Doc 1) but missed `10 Things I Hate About You` (Doc 2, which mentions the 1999 movie co-starring Heath Ledger and Julia Stiles). Without Doc 2, the LLM extracted `2016` (the release year of *Snowden*) from Doc 1.

---

### Question 8: Index 23 (`5ab985b5554299131ca4235e`)
- **Question:** *The Crystal Palace Football Club Player of the Year is awarded at the end of each season... most recent being which professional footballer, who plays as a winger for Premier League club Crystal Palace, and the Ivory Coast national team?*
- **Gold Answer:** `Wilfried Zaha`
- **RAG Generated Answer:** `The relevant information to answer the above question is: Dazet Wilfried Armel Zaha (born 10 November 1992) is a professional footballer who plays as a winger for Premier League club Crystal Palace and the Ivory`
- **Failure Category:** `Generation Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved**

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Crystal Palace F.C. Player of the Year | `54185106` | 0.6959 | **Yes** | Nine of these players have won the award for a second time, the most recent being Wilfried Zaha... |
| 2 | Wilfried Zaha | `28871495` | 0.5056 | **Yes** | Dazet Wilfried Armel Zaha... plays as a winger for Premier League club Crystal Palace and the Ivory Coast... |
| 3 | Corey Davis (wide receiver) | `51406206` | 0.2677 | No | Corey Davis... American football wide receiver... |

- **Root Cause Analysis:** Both gold documents were retrieved. The model contained the correct entity (`Wilfried Zaha` / `Dazet Wilfried Armel Zaha`), but repeated the instruction prefix and ran out of output tokens before formatting the concise entity name.

---

### Question 9: Index 31 (`5ab8f1fa5542991b5579f077`)
- **Question:** *Are Gary Lightbody and Dustin Kensrue from the same country?*
- **Gold Answer:** `no`
- **RAG Generated Answer:** `yes`
- **Failure Category:** `Multi-Hop Reasoning Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved** (100% ground-truth context available)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Dustin Kensrue | `2893773` | 0.5779 | **Yes** | Dustin Michael Kensrue... born in Orange County, California... |
| 2 | Gary Lightbody | `6131604` | 0.4624 | **Yes** | Gary Lightbody... is a Northern Irish singer... lead singer of Northern Irish-Scottish rock band Snow Patrol... |
| 3 | Robert Evans (photographer) | `39928514` | 0.2247 | No | Robert Evans... American photographer... |

- **Root Cause Analysis:** Both documents were in context (`Gary Lightbody` $\rightarrow$ Northern Ireland; `Dustin Kensrue` $\rightarrow$ USA / California). FLAN-T5-Base failed geographical reasoning and predicted `yes` instead of `no`.

---

### Question 10: Index 34 (`5ae7ac1f5542993210983eed`)
- **Question:** *What is the name of the Los Angeles-based production company that produced a film based on the novel by André Aciman?*
- **Gold Answer:** `Water's End Productions`
- **RAG Generated Answer:** `Luca Guadagnino`
- **Failure Category:** `Retrieval Failure`
- **Relevant Gold Document Retrieval Status:** **1/2 Retrieved** (Missing Document: `Tom Dolby`)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Assassins (film) | `3300211` | 0.3298 | No | Assassins is a 1995 American action thriller... |
| 2 | Sacha Baron Cohen | `45218` | 0.3128 | No | Sacha Noam Baron Cohen... |
| 3 | Call Me by Your Name (film) | `50731291` | 0.2882 | **Yes** | Call Me by Your Name... directed by Luca Guadagnino... based on the novel of the same name by André Aciman... |

- **Root Cause Analysis:** Dense retrieval fetched `Call Me by Your Name (film)` (Doc 1) but missed `Tom Dolby` (Doc 2, which identifies `Water's End Productions` as the Los Angeles-based production company). The LLM extracted director `Luca Guadagnino` from Doc 1.

---

### Question 11: Index 35 (`5a74691755429929fddd8412`)
- **Question:** *What friends star's wedding was photographed by Robert Evans?*
- **Gold Answer:** `Jennifer Aniston`
- **RAG Generated Answer:** `Brad Pitt`
- **Failure Category:** `Retrieval Failure`
- **Relevant Gold Document Retrieval Status:** **1/2 Retrieved** (Missing Document: `Jennifer Aniston`)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Robert Evans (photographer) | `39928514` | 0.7690 | **Yes** | ...celebrity wedding photographer, which includes the weddings of Brad Pitt and Jennifer Aniston and Tom Cruise... |
| 2 | Titus Davis | `46521626` | 0.2449 | No | Titus Davis... wide receiver... |
| 3 | Candleshoe | `2225976` | 0.2407 | No | Candleshoe is a 1977 film... |

- **Root Cause Analysis:** Retrieval fetched `Robert Evans` (Doc 1) listing weddings of Brad Pitt and Jennifer Aniston. However, it missed `Jennifer Aniston` (Doc 2, which states she starred in *Friends*). The model picked `Brad Pitt` from Doc 1.

---

### Question 12: Index 36 (`5a8dea555542995a26add44f`)
- **Question:** *What is the prequel to the show where Rose Eleanor Arbuthnot-Leslie stars as Maia Rindell?*
- **Gold Answer:** `The Good Wife"`
- **RAG Generated Answer:** `The Good Fight`
- **Failure Category:** `Retrieval Failure`
- **Relevant Gold Document Retrieval Status:** **1/2 Retrieved** (Missing Document: `The Good Fight`)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Rose Leslie | `29540174` | 0.5364 | **Yes** | Rose Eleanor Arbuthnot-Leslie... stars as Maia Rindell in the CBS All Access legal drama "The Good Fight"... |
| 2 | Candleshoe | `2225976` | 0.3291 | No | Candleshoe... |
| 3 | All Quiet on the Western Front | `4592959` | 0.2197 | No | All Quiet on the Western Front... |

- **Root Cause Analysis:** Retrieval fetched `Rose Leslie` (Doc 1) which names her show as *The Good Fight*. It missed `The Good Fight` (Doc 2, which states *The Good Fight* is the sequel to *The Good Wife*). The LLM outputted the show title (*The Good Fight*) rather than its prequel (*The Good Wife*).

---

### Question 13: Index 40 (`5ade08425542995b365fabd9`)
- **Question:** *University of Toronto and Istanbul Technical University, are located in Canada?*
- **Gold Answer:** `no`
- **RAG Generated Answer:** `yes`
- **Failure Category:** `Multi-Hop Reasoning Failure`
- **Relevant Gold Document Retrieval Status:** **2/2 Retrieved** (100% ground-truth context available)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | University of Toronto | `7955325` | 0.5984 | **Yes** | The University of Toronto... is a public research university in Toronto, Ontario, Canada... |
| 2 | Istanbul Technical University | `743431` | 0.5619 | **Yes** | Istanbul Technical University... is an international technical university located in Istanbul, Turkey... |
| 3 | Indian Statistical Institute | `2238418` | 0.2719 | No | Indian Statistical Institute... |

- **Root Cause Analysis:** Both documents were retrieved. FLAN-T5-Base observed that the University of Toronto is in Canada, but failed boolean conjunction logic ($\text{Canada} \land \text{Turkey} \implies \text{not both in Canada}$) and answered `yes`.

---

### Question 14: Index 43 (`5add1ebe5542990d50227dec`)
- **Question:** *Who was the director of the 2009 British thriller film starring the actor who is best known for playing Robert Crawley in "Downtown Abbey"?*
- **Gold Answer:** `Anthony Hickox`
- **RAG Generated Answer:** `Coky Giedroyc`
- **Failure Category:** `Retrieval Failure`
- **Relevant Gold Document Retrieval Status:** **1/2 Retrieved** (Missing Document: `Knife Edge (film)`)

#### Top-3 Retrieved Chunks ($K=3$)
| Rank | Title | Doc ID | Cosine Score | Relevant Gold? | Snippet |
|---|---|---|---|---|---|
| 1 | Hugh Bonneville | `1802353` | 0.6630 | **Yes** | Hugh Richard Bonneville Williams... best known for playing Robert Crawley in... "Downton Abbey"... |
| 2 | Wuthering Heights (2009 TV serial) | `20722922` | 0.3649 | No | Wuthering Heights is a two-part ITV television series adaptation... |
| 3 | Joseph Gordon-Levitt | `1817766` | 0.3405 | No | Joseph Leonard Gordon-Levitt... |

- **Root Cause Analysis:** Vector search retrieved `Hugh Bonneville` (Doc 1) but missed `Knife Edge (film)` (Doc 2, which identifies `Anthony Hickox` as director). FLAN-T5-Base hallucinated director `Coky Giedroyc`.

---

## 4. Key Engineering Insights & Future RAG Improvements

1. **Dense Retrieval Bottleneck (42.9% of errors):**
   - Single-vector embedding models (`all-MiniLM-L6-v2`) match semantic similarity to query keywords but frequently fail to retrieve the **2nd-hop entity bridge document** when its title or content shares low direct lexical overlap with the query.
   - *Fix:* Implement **BM25 + Dense Hybrid Search**, **Query Rewriting**, or **Cross-Encoder Reranking**.

2. **LLM Multi-Hop Reasoning Deficit (35.7% of errors):**
   - Small models like FLAN-T5-Base (250M parameters) struggle with logical operations across multiple text snippets (e.g., temporal ordering, set membership, boolean conjunctions, and voice actor entity bindings).
   - *Fix:* Use structured instruction prompt templates or evaluate larger model backbones (e.g., FLAN-T5-XL, Llama-3-8B).

3. **Prompt Prefix & Formatting Artifacts (14.3% of errors):**
   - The generator prompt template caused FLAN-T5 to echo instructional prefix text (`"The relevant information to answer..."`).
   - *Fix:* Refine prompt formatting to enforce direct, concise entity extraction (e.g., `"Answer in 1-3 words strictly:"`).
