# 🚀 Baseline RAG Pipeline Experiment Report

**Generated:** 2026-10-03T15:49:13  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Vector Index:** `FAISS IndexFlatIP` (Cosine Similarity)  
**Generator Model:** `google/flan-t5-base` (Greedy, Temp=0.0, Seed=42)  
**Top-K Retrieval:** `k = 3`  
**Total Execution Time:** `125.22s`  

---

## 📊 Key Experimental Findings & Baseline Comparison

| Benchmark Baseline | Correct Rate | Exact Match (EM) | Retrieval Hit Rate @ 3 | Retrieval Recall @ 3 |
| :--- | :---: | :---: | :---: | :---: |
| **Closed-Book Baseline** (No Retrieval) | `10.0%` (5/50) | `10.0%` | N/A | N/A |
| **Baseline RAG Pipeline** (Open-Book) | **`72.0%` (36/50)** | **`58.0%`** | **`100.0%`** | **`0.93`** |
| **Net Performance Gain** | **`+62.0%`** | **`+48.0%`** | — | — |

> **Key Result:** Adding vector retrieval boosted closed-book accuracy from **10.0%** to **72.0%** (a **+62.0% absolute gain**), demonstrating the power of open-book context augmentation!

---

## ⚙️ Complete Experiment Configuration

```json
{
  "experiment_name": "Baseline RAG Pipeline",
  "random_seed": 42,
  "dataset_info": {
    "queries_count": 50,
    "corpus_documents_count": 100,
    "qrels_count": 100
  },
  "chunking_settings": {
    "chunk_size_words": 250,
    "chunk_overlap_words": 50,
    "total_chunks_created": 100
  },
  "retrieval_settings": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_index_type": "FAISS IndexFlatIP (Cosine Similarity)",
    "top_k": 3,
    "indexing_time_seconds": 2.169
  },
  "generation_settings": {
    "generator_model": "google/flan-t5-base",
    "do_sample": false,
    "temperature": 0.0,
    "max_new_tokens": 50,
    "prompt_template": "Context:\n{context}\n\nQuestion: {question}\nAnswer the question based ONLY on the context provided above:"
  },
  "advanced_features": {
    "reranking": false,
    "query_rewriting": false,
    "hybrid_search": false
  },
  "runtime_seconds": 125.22
}
```

---

## 🔍 Retrieval & Generation Per-Question Results

| # | Query ID | Question | Top-1 Retrieved Doc & Score | RAG Answer | Gold Answer | Assessment |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: |
| 1 | `5a7c35885542996dd594b8d9` | The Hope Theatre presented plays during what overall time frame in England? | `7629569` (Hope Theatre) [0.763] | `between 1562 and 1642` | `1562 and 1642` | ✅ Correct |
| 2 | `5a8e4a605542995a26add4cb` | The composer of the opera "The Zoo" has composed how many major orchestral works? | `4934253` (The Zoo) [0.671] | `13` | `13` | ✅ Correct |
| 3 | `5a8780385542996e4f30880e` | What character in the television special "The Pound Puppies" did the voice actor of Fred Jones from "Scooby-Doo" voice as? | `487732` (Frank Welker) [0.659] | `Cooler` | `Howler` | ❌ Incorrect |
| 4 | `5a8cb59055429941ae14ded6` | What is the name of the television series "Wuthering Heights" actor who also starred in the action film "Black Hawk Down"? | `20722922` (Wuthering Heights (2009 TV serial)) [0.603] | `Tom Hardy` | `Tom Hardy` | ✅ Correct |
| 5 | `5ae5fb975542996de7b71aa8` | Edmundo "Eddie" Rodriguez was born in what Mexican state? | `21757108` (Edmundo Rodriguez) [0.827] | `Guadalajara` | `Jalisco` | ❌ Incorrect |
| 6 | `5abd0e8355429924427fcee2` | Sylvia Sidney starred in the film Beetlejuice, which was directed by whom? | `1046971` (Sylvia Sidney) [0.758] | `Tim Burton` | `Tim Burton` | ✅ Correct |
| 7 | `5a7257ce5542990c210a410d` | This English comedian wrote the British sitcom "Absolutely Fabulous" | `49546885` (Absolutely Fabulous (series 3)) [0.565] | `John Sullivan` | `Jennifer Saunders` | ❌ Incorrect |
| 8 | `5ab739c65542993667793f9a` | "The Class of '62" is an episode of the BBC television sitcom, "Only Fools and Horses", created and written by who? | `7148380` (The Class of '62) [0.871] | `John Sullivan` | `John Sullivan` | ✅ Correct |
| 9 | `5a8c5d795542995e66a475d4` | Where is the company located that produced the 2006 James Bond movie Casino Royale? | `930379` (Casino Royale (2006 film)) [0.679] | `London's Piccadilly` | `based in London's Piccadilly and also operates from Pinewood Studios in the United Kingdom` | ✅ Correct |
| 10 | `5add443a5542992ae4cec503` | What American biotechnology company is part of Gloucester Pharmaceuticals? | `2887610` (Celgene) [0.373] | `Celgene Corporation` | `Celgene` | ✅ Correct |
| 11 | `5a8b2d8e5542995d1e6f12f1` | Which hall at the University of California school, which also has a branch in Palm Desert, is named after the first chancellor of this school? | `230311` (University of California, Riverside) [0.562] | `Spieth Hall` | `Spieth Hall` | ✅ Correct |
| 12 | `5ac1d63055429964131be2a1` | E-mails from Bill Clinton's Chief of staff lead to the pizzagate conspiracy in what year? | `52376302` (Pizzagate conspiracy theory) [0.738] | `2016` | `2016` | ✅ Correct |
| 13 | `5adf48b25542995ec70e8fb5` | In what city did Amir Khan and Lamont Peterson fight for the welterweight titles in 2011? | `7558034` (Lamont Peterson) [0.736] | `London` | `British` | ❌ Incorrect |
| 14 | `5a89c2715542993b751ca990` | Which was released first, God's Angry Man or The World's Best Prom? | `3751418` (The World's Best Prom) [0.573] | `The World's Best Prom` | `God's Angry Man` | ❌ Incorrect |
| 15 | `5a7612cf55429976ec32bcfe` | Which American wide receiver has a brother who was drafted by the Titans fifth overall in the 2017 NFL Draft? | `46521626` (Titus Davis) [0.666] | `Titus Davis` | `Titus Davis` | ✅ Correct |
| 16 | `5ab9c21055429970cfb8eb9d` | Michael Jones is a competitor in sport where motorcycle riders attempt to impress judges with what? | `15838235` (Mike Jones (motocross rider)) [0.613] | `jumps and stunts` | `jumps and stunts` | ✅ Correct |
| 17 | `5a7e264055429965cec5eaa4` | Fulda Cathedral is the burial place of the leading figure in what? | `28227842` (Fulda Cathedral) [0.768] | `The relevant information to answer the above question is: Fulda Cathedral (German: "Fuldaer Dom" , also "Sankt Salvator") is the former abbey church of Fulda Abbey and the` | `Anglo-Saxon mission` | ❌ Incorrect |
| 18 | `5a88a74a554299206df2b31b` | What profession does Lewis Milestone and All Quiet on the Western Front have in common? | `62262` (Lewis Milestone) [0.487] | `motion picture director` | `director` | ✅ Correct |
| 19 | `5ae1bb88554299234fd042dd` | Are both Candleshoe and Pooh's Heffalump Movie Walt Disney films? | `1499129` (Pooh's Heffalump Movie) [0.651] | `yes` | `yes` | ✅ Correct |
| 20 | `5ae4d3235542993aec5ec0dd` | Lorilee Craker co-authored a memoir about a woman who is what relation to Britney Spears? | `46928473` (Lorilee Craker) [0.703] | `mother` | `mother` | ✅ Correct |
| 21 | `5a71532e5542994082a3e7d6` | In what year did the actor who played Edward Snowden act in a movie with Heath Ledger and Julia Styles? | `1817766` (Joseph Gordon-Levitt) [0.417] | `2016` | `1999` | ❌ Incorrect |
| 22 | `5ae5e5fc55429929b0807a06` | John Shakespeare was the son of the husbandman who lived in what Warwickshire town? | `7672206` (Richard Shakespeare) [0.812] | `Snitterfield` | `Snitterfield` | ✅ Correct |
| 23 | `5ab985b5554299131ca4235e` | The Crystal Palace Football Club Player of the Year is awarded at the end of each season, nine of these players have won the award for a second time, the most recent being which professional footballer, who plays as a winger for Premier League club Crystal Palace, and the Ivory Coast national team? | `54185106` (Crystal Palace F.C. Player of the Year) [0.696] | `The relevant information to answer the above question is: Dazet Wilfried Armel Zaha (born 10 November 1992) is a professional footballer who plays as a winger for Premier League club Crystal Palace and the Ivory` | `Wilfried Zaha` | ❌ Incorrect |
| 24 | `5abef2505542993fe9a41db2` | Ulrich Matthes is best known for his role in  a 2004 German-Italian-Austrian historical war drama film directed by who? | `4255524` (Ulrich Matthes) [0.640] | `Oliver Hirschbiegel` | `Oliver Hirschbiegel` | ✅ Correct |
| 25 | `5a8e0c34554299068b959e37` | The Tremont Stakes in June is among the classic races held at Belmont Park in which U.S. state? | `9352902` (Tremont Stakes) [0.873] | `New York` | `New York` | ✅ Correct |
| 26 | `5a87dc635542994846c1cdf2` | When was the Polish politician and former Vice-Minister of Foreign Affairs of Poland, who was a Doctor of Law, and who candidated from a right-wing populist, national-conservative, and Christian democratic political party in Poland, elected to the Sejm? | `5814825` (Karol Karski) [0.725] | `September 25, 2005` | `September 25, 2005` | ✅ Correct |
| 27 | `5a81de8d55429926c1cdada9` | Which English actor co-wrote and also played in The Dictator? | `30553238` (The Dictator (2012 film)) [0.577] | `Sacha Baron Cohen` | `Sacha Baron Cohen` | ✅ Correct |
| 28 | `5a7559b25542992db9473648` | Nupedia was underwritten by what company that was founded in 1996? | `26155570` (Nupedia) [0.763] | `Bomis` | `Bomis` | ✅ Correct |
| 29 | `5a8b634255429949d91db58b` | What symbol did a Ghanaian stateswoman, teacher and artist known for designing Ghana's flag in 1957 use in that flag? | `40343708` (Theodosia Okoh) [0.657] | `the black star` | `The Black Star of Africa` | ✅ Correct |
| 30 | `5a7f81bc5542992097ad2f8f` | Sanghamitra Bandyopadhyay a professor at theIndian Statistical Institute, a public university established in what year? | `44152621` (Sanghamitra Bandyopadhyay) [0.761] | `1931` | `1931` | ✅ Correct |
| 31 | `5ab8f1fa5542991b5579f077` | Are Gary Lightbody and Dustin Kensrue from the same country? | `2893773` (Dustin Kensrue) [0.578] | `yes` | `no` | ❌ Incorrect |
| 32 | `5a74b6ea55429974ef308c41` | What commune, ecovillage, and spiritual community was founded by Oberto Airaudi? | `8454308` (Oberto Airaudi) [0.572] | `Federation of Damanhur` | `Federation of Damanhur` | ✅ Correct |
| 33 | `5ab709b15542992aa3b8c78e` | The writers of the "Assassins" screenplay made their directing debut in which year? | `3300211` (Assassins (film)) [0.808] | `1996` | `1996` | ✅ Correct |
| 34 | `5ae7ac1f5542993210983eed` | What is the name of the Los Angeles-based production company that produced a film based on the novel by André Aciman? | `3300211` (Assassins (film)) [0.330] | `Luca Guadagnino` | `Water's End Productions` | ❌ Incorrect |
| 35 | `5a74691755429929fddd8412` | What friends star's wedding was photographed by Robert Evans? | `39928514` (Robert Evans (photographer)) [0.769] | `Brad Pitt` | `Jennifer Aniston` | ❌ Incorrect |
| 36 | `5a8dea555542995a26add44f` | What is the prequel to the show where Rose Eleanor Arbuthnot-Leslie stars as Maia Rindell? | `29540174` (Rose Leslie) [0.536] | `The Good Fight` | `The Good Wife"` | ❌ Incorrect |
| 37 | `5ab4826f5542990594ba9c3c` | When was Justin Vincent the MVP of a collegiate championship game? | `4892221` (Justin Vincent) [0.607] | `2004` | `January 4, 2004` | ✅ Correct |
| 38 | `5a7735ac5542993735360219` | Which actor starred in Assignment to Kill and passed away in 2000. | `23066976` (Assignment to Kill) [0.651] | `John Gielgud` | `John Gielgud` | ✅ Correct |
| 39 | `5ab2bfc65542991669774094` | When did the British/Irish digital television channel, which features Spliced, first launch? | `24403156` (Spliced (TV series)) [0.637] | `22 July 2002` | `22 July 2002` | ✅ Correct |
| 40 | `5ade08425542995b365fabd9` | University of Toronto and Istanbul Technical University, are located in Canada? | `7955325` (University of Toronto) [0.598] | `yes` | `no` | ❌ Incorrect |
| 41 | `5ac5093b5542996feb3fe9d8` | Plymouth Regional High School is located in what county? | `2962586` (Plymouth Regional High School (New Hampshire)) [0.717] | `Grafton County` | `Grafton County` | ✅ Correct |
| 42 | `5ab26ed65542993be8fa9913` | Does the Inula genus or the Fouquieria genus have more species? | `1906719` (Inula) [0.695] | `Inula` | `Inula` | ✅ Correct |
| 43 | `5add1ebe5542990d50227dec` | Who was the director of the 2009 British thriller film starring the actor who is best known for playing Robert Crawley in "Downtown Abbey"? | `1802353` (Hugh Bonneville) [0.663] | `Coky Giedroyc` | `Anthony Hickox` | ❌ Incorrect |
| 44 | `5a7541ab5542993748c89803` | What school is located in the southernmost atoll of the archipelago of the Maldives? | `898257` (Addu City) [0.530] | `Maradhoo Feydhoo School` | `Maradhoo Feydhoo School` | ✅ Correct |
| 45 | `5a83e925554299123d8c21ca` | Kofi Dzamesi (born 1960) is a Ghanaian engineer and politician, since February 2017 he has served as Minister of which official Ghanaian agency responsible the creation of linkages between the Government of Ghana and the traditional authorities in the country? | `54682964` (Kofi Dzamesi) [0.903] | `The Ministry for Chieftaincy and Traditional Affairs` | `Ministry for Chieftaincy and Traditional Affairs` | ✅ Correct |
| 46 | `5a7a783b5542990783324f58` | What country of origin does 17 Carat and Seventeen have in common? | `52427290` (17 Carat) [0.595] | `South Korean` | `Korean` | ✅ Correct |
| 47 | `5ae07f6d55429945ae959385` | Who is the protagonist in Ayn Rand's most notable novel? | `22256` (Objectivism (Ayn Rand)) [0.656] | `Howard Roark` | `Howard Roark` | ✅ Correct |
| 48 | `5ab2f57955429929539468cf` | The composer of A Survivor from Warsaw emigrated to the United States in what year? | `6566239` (A Survivor from Warsaw) [0.693] | `1934` | `1934` | ✅ Correct |
| 49 | `5a7def2a5542995f4f402324` | Anna Howard Shaw Day starred the actor, director, and producer who was best known for playing what character on Mad Men? | `26171562` (Anna Howard Shaw Day) [0.537] | `Don Draper` | `Don Draper` | ✅ Correct |
| 50 | `5ae2e2c655429928c423952d` | When was the American Democratic politician which was the governor of 185th New York State Legislature born? | `45604736` (185th New York State Legislature) [0.660] | `June 15, 1932` | `June 15, 1932` | ✅ Correct |
