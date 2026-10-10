# 📊 Linked HotpotQA Pilot Validation Report

**Generated:** 2026-10-03T14:23:20.338980  
**Dataset Source:** `BeIR/hotpotqa` & `BeIR/hotpotqa-qrels`  
**Manifest Path:** `data/manifest.json`  

---

## 1. Executive Summary & Verification Confirmation

> **Full Coverage Confirmation:**  
> **✅ PASSED:** Every single selected query (50 queries) has **100%** of its ground-truth relevant documents (100 corpus documents) present in the local pilot corpus. Zero missing documents detected.

---

## 2. Dataset Metrics Overview

| Metric | Count / Value | Status / Note |
| :--- | :---: | :--- |
| **Total Queries** | `50` | 50 sampled queries |
| **Total Corpus Documents** | `100` | 100 linked Wikipedia passages |
| **Total Qrels Links** | `100` | 100 relevance pairs (score = 1) |
| **Relevant Docs / Query (Mean)** | `2.0` | Exactly 2 relevant docs per query |
| **Duplicate Query IDs** | `0` | Clean (No duplicates) |
| **Duplicate Corpus IDs** | `0` | Clean (No duplicates) |
| **Duplicate Qrel Pairs** | `0` | Clean (No duplicates) |
| **Missing Query IDs** | `0` | All Qrel queries exist in queries pilot |
| **Missing Corpus Document IDs** | `0` | All Qrel documents exist in corpus pilot |
| **Empty Query Texts** | `0` | Clean (No empty queries) |
| **Empty Corpus Passages** | `0` | Clean (No empty passages) |
| **Empty Corpus Titles** | `0` | Clean (All passages titled) |

---

## 3. Relevant Documents per Query Distribution

```text
Relevant Docs per Query:
  - Exactly 2 documents per query: 50 queries (100.0%)
```

---

## 4. Query Length Statistics

| Statistic | Character Length | Word Length |
| :--- | :---: | :---: |
| **Mean** | `98.34` | `16.26` |
| **Std Dev** | `49.69` | `8.34` |
| **Minimum** | `56` | `9` |
| **25th Percentile (P25)** | `66.25` | `11.0` |
| **Median (P50)** | `83.5` | `14.0` |
| **75th Percentile (P75)** | `113.5` | `19.75` |
| **Maximum** | `299` | `53` |

---

## 5. 10 Sample Query – Relevant-Document Pairs

### Sample 1: Query `5a7c35885542996dd594b8d9`
**Question:** The Hope Theatre presented plays during what overall time frame in England?

**Relevant Documents:**
- **Doc ID:** `7629569` | **Title:** `Hope Theatre`
  - *Snippet:* The Hope Theatre was one of the theatres built in and around London for the presentation of plays in English Renaissance theatre, comparable to the Globe, the Curtain, the Swan, an...
- **Doc ID:** `39566` | **Title:** `English Renaissance theatre`
  - *Snippet:* English Renaissance theatre, also known as early modern English theatre, or (commonly) as Elizabethan theatre, refers to the theatre of England between 1562 and 1642....

### Sample 2: Query `5a8e4a605542995a26add4cb`
**Question:** The composer of the opera "The Zoo" has composed how many major orchestral works?

**Relevant Documents:**
- **Doc ID:** `4934253` | **Title:** `The Zoo`
  - *Snippet:* The Zoo is a one-act comic opera, with music by Arthur Sullivan and a libretto by B. C. Stephenson, writing under the pen name of Bolton Rowe. It premiered on 5 June 1875 at the St...
- **Doc ID:** `45280` | **Title:** `Arthur Sullivan`
  - *Snippet:* Sir Arthur Seymour Sullivan MVO (13 May 1842 – 22 November 1900) was an English composer. He is best known for his series of 14 operatic collaborations with the dramatist W. S. Gil...

### Sample 3: Query `5a8780385542996e4f30880e`
**Question:** What character in the television special "The Pound Puppies" did the voice actor of Fred Jones from "Scooby-Doo" voice as?

**Relevant Documents:**
- **Doc ID:** `37328049` | **Title:** `Pound Puppies (film)`
  - *Snippet:* The Pound Puppies is an animated television special, produced by Hanna-Barbera Productions, based on the popular toy line from Tonka, which aired in syndication on October 26, 1985...
- **Doc ID:** `487732` | **Title:** `Frank Welker`
  - *Snippet:* Franklin Wendell Welker (born March 12, 1946) is an American actor, who for most of his career has specialized as a voice actor. He is best known for his role as Fred Jones from th...

### Sample 4: Query `5a8cb59055429941ae14ded6`
**Question:** What is the name of the television series "Wuthering Heights" actor who also starred in the action film "Black Hawk Down"?

**Relevant Documents:**
- **Doc ID:** `20722922` | **Title:** `Wuthering Heights (2009 TV serial)`
  - *Snippet:* Wuthering Heights is a two-part ITV television series adaptation of the novel "Wuthering Heights" by Emily Brontë. The episodes were adapted for the screen by Peter Bowker and dire...
- **Doc ID:** `1792975` | **Title:** `Tom Hardy`
  - *Snippet:* (born 15 September 1977) is an English actor and producer. His motion picture debut was in Ridley Scott's 2001 action film "Black Hawk Down". Hardy's other notable films include th...

### Sample 5: Query `5ae5fb975542996de7b71aa8`
**Question:** Edmundo "Eddie" Rodriguez was born in what Mexican state?

**Relevant Documents:**
- **Doc ID:** `21757108` | **Title:** `Edmundo Rodriguez`
  - *Snippet:* Edmundo "Eddie" Rodriguez (born April 22, 1970 in Guadalajara, Mexico) is a retired Mexican soccer player. He spent most of his career in the U.S. leagues including two seasons in ...
- **Doc ID:** `73209` | **Title:** `Guadalajara`
  - *Snippet:* Guadalajara ( , ] ) is the capital and largest city of the Mexican state of Jalisco, and the seat of the municipality of Guadalajara. The city is in the central region of Jalisco i...

### Sample 6: Query `5abd0e8355429924427fcee2`
**Question:** Sylvia Sidney starred in the film Beetlejuice, which was directed by whom?

**Relevant Documents:**
- **Doc ID:** `1046971` | **Title:** `Sylvia Sidney`
  - *Snippet:* Sylvia Sidney (born Sophia Kosow; August 8, 1910 – July 1, 1999) was an American actress of stage, screen and film, with a career spanning over 70 years, who first rose to prominen...
- **Doc ID:** `293524` | **Title:** `Beetlejuice`
  - *Snippet:* Beetlejuice is a 1988 American comedy-fantasy film directed by Tim Burton, produced by The Geffen Film Company and distributed by Warner Bros. The plot revolves around a recently d...

### Sample 7: Query `5a7257ce5542990c210a410d`
**Question:** This English comedian wrote the British sitcom "Absolutely Fabulous"

**Relevant Documents:**
- **Doc ID:** `49546885` | **Title:** `Absolutely Fabulous (series 3)`
  - *Snippet:* The third series of "Absolutely Fabulous", a British sitcom, premiered on BBC One on 30 March 1995. This series consists of six episodes and concluded on 11 May 1995. Initially, th...
- **Doc ID:** `89605` | **Title:** `Jennifer Saunders`
  - *Snippet:* Jennifer Jane Saunders (born 6 July 1958) is an English comedian, screenwriter, and actress. She has won three BAFTAs (including the BAFTA Fellowship), an International Emmy Award,...

### Sample 8: Query `5ab739c65542993667793f9a`
**Question:** "The Class of '62" is an episode of the BBC television sitcom, "Only Fools and Horses", created and written by who?

**Relevant Documents:**
- **Doc ID:** `7148380` | **Title:** `The Class of '62`
  - *Snippet:* "The Class of '62" is an episode of the BBC sit-com "Only Fools and Horses". It was the fourth episode of series 7, and was first broadcast on 20 January 1991. In the episode, Del ...
- **Doc ID:** `200842` | **Title:** `Only Fools and Horses`
  - *Snippet:* Only Fools and Horses is a British television sitcom created and written by John Sullivan. Seven series were originally broadcast on BBC One in the United Kingdom from 1981 to 1991...

### Sample 9: Query `5a8c5d795542995e66a475d4`
**Question:** Where is the company located that produced the 2006 James Bond movie Casino Royale?

**Relevant Documents:**
- **Doc ID:** `930379` | **Title:** `Casino Royale (2006 film)`
  - *Snippet:* Casino Royale (2006) is the twenty-first spy film in the Eon Productions "James Bond" film series and the first to star Daniel Craig as the fictional MI6 agent James Bond. Directed...
- **Doc ID:** `726979` | **Title:** `Eon Productions`
  - *Snippet:* Eon Productions (an abbreviation of Everything or Nothing) is a British film production company that produces the "James Bond" film series. The company is based in London's Piccadi...

### Sample 10: Query `5add443a5542992ae4cec503`
**Question:** What American biotechnology company is part of Gloucester Pharmaceuticals?

**Relevant Documents:**
- **Doc ID:** `24314125` | **Title:** `Romidepsin`
  - *Snippet:* Romidepsin, also known as Istodax, is an anticancer agent used in cutaneous T-cell lymphoma (CTCL) and other peripheral T-cell lymphomas (PTCLs). Romidepsin is a natural product ob...
- **Doc ID:** `2887610` | **Title:** `Celgene`
  - *Snippet:* Celgene Corporation is an American biotechnology company that discovers, develops and commercializes medicines for cancer and inflammatory disorders. It is incorporated in Delaware...

