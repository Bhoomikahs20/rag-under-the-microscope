# 🤖 Closed-Book Evaluation Report: HotpotQA Pilot

**Evaluation Type:** Closed-Book QA (No Context / No Retrieval / No External Search)  
**Model:** `google/flan-t5-base` (Greedy Decoding, Temp=0.0, Seed=42)  
**Total Queries:** `50`  
**Overall Correct Rate:** **`10.0%`** (`5/50` correct)  
**Exact Match (EM) Rate:** `10.0%` (`5/50`)  
**Total Eval Time:** `25.29s`  

---

## ⚙️ Model & Configuration Details

```json
{
  "model_id": "google/flan-t5-base",
  "architecture": "Seq2Seq / Flan-T5-Base",
  "parameters": "250M",
  "closed_book_prompt_template": "Answer the question: {question}",
  "decoding_parameters": {
    "do_sample": false,
    "temperature": 0.0,
    "max_new_tokens": 50,
    "random_seed": 42
  },
  "retrieval_used": false,
  "corpus_access_allowed": false,
  "external_search_allowed": false
}
```

---

## 📋 Evaluation Results by Question

| # | Query ID | Question | Model Answer (Closed-Book) | Gold Answer | Assessment |
| :-: | :--- | :--- | :--- | :--- | :-: |
| 1 | `5a7c35885542996dd594b8d9` | The Hope Theatre presented plays during what overall time frame in England? | `mid-19th century` | `1562 and 1642` | ❌ Incorrect |
| 2 | `5a8e4a605542995a26add4cb` | The composer of the opera "The Zoo" has composed how many major orchestral works? | `seven` | `13` | ❌ Incorrect |
| 3 | `5a8780385542996e4f30880e` | What character in the television special "The Pound Puppies" did the voice actor of Fred Jones from "Scooby-Doo" voice as? | `a pig` | `Howler` | ❌ Incorrect |
| 4 | `5a8cb59055429941ae14ded6` | What is the name of the television series "Wuthering Heights" actor who also starred in the action film "Black Hawk Down"? | `john dillinger` | `Tom Hardy` | ❌ Incorrect |
| 5 | `5ae5fb975542996de7b71aa8` | Edmundo "Eddie" Rodriguez was born in what Mexican state? | `san pedro` | `Jalisco` | ❌ Incorrect |
| 6 | `5abd0e8355429924427fcee2` | Sylvia Sidney starred in the film Beetlejuice, which was directed by whom? | `john mccartney` | `Tim Burton` | ❌ Incorrect |
| 7 | `5a7257ce5542990c210a410d` | This English comedian wrote the British sitcom "Absolutely Fabulous" | `john mccartney` | `Jennifer Saunders` | ❌ Incorrect |
| 8 | `5ab739c65542993667793f9a` | "The Class of '62" is an episode of the BBC television sitcom, "Only Fools and Horses", created and written by who? | `john mccartney` | `John Sullivan` | ❌ Incorrect |
| 9 | `5a8c5d795542995e66a475d4` | Where is the company located that produced the 2006 James Bond movie Casino Royale? | `New York City` | `based in London's Piccadilly and also operates from Pinewood Studios in the United Kingdom` | ❌ Incorrect |
| 10 | `5add443a5542992ae4cec503` | What American biotechnology company is part of Gloucester Pharmaceuticals? | `GE` | `Celgene` | ❌ Incorrect |
| 11 | `5a8b2d8e5542995d1e6f12f1` | Which hall at the University of California school, which also has a branch in Palm Desert, is named after the first chancellor of this school? | `san diego` | `Spieth Hall` | ❌ Incorrect |
| 12 | `5ac1d63055429964131be2a1` | E-mails from Bill Clinton's Chief of staff lead to the pizzagate conspiracy in what year? | `1998` | `2016` | ❌ Incorrect |
| 13 | `5adf48b25542995ec70e8fb5` | In what city did Amir Khan and Lamont Peterson fight for the welterweight titles in 2011? | `san francisco` | `British` | ❌ Incorrect |
| 14 | `5a89c2715542993b751ca990` | Which was released first, God's Angry Man or The World's Best Prom? | `The World's Best Prom` | `God's Angry Man` | ❌ Incorrect |
| 15 | `5a7612cf55429976ec32bcfe` | Which American wide receiver has a brother who was drafted by the Titans fifth overall in the 2017 NFL Draft? | `Jason Campbell` | `Titus Davis` | ❌ Incorrect |
| 16 | `5ab9c21055429970cfb8eb9d` | Michael Jones is a competitor in sport where motorcycle riders attempt to impress judges with what? | `speed` | `jumps and stunts` | ❌ Incorrect |
| 17 | `5a7e264055429965cec5eaa4` | Fulda Cathedral is the burial place of the leading figure in what? | `scottish king scott` | `Anglo-Saxon mission` | ❌ Incorrect |
| 18 | `5a88a74a554299206df2b31b` | What profession does Lewis Milestone and All Quiet on the Western Front have in common? | `novelist` | `director` | ❌ Incorrect |
| 19 | `5ae1bb88554299234fd042dd` | Are both Candleshoe and Pooh's Heffalump Movie Walt Disney films? | `yes` | `yes` | ✅ Correct |
| 20 | `5ae4d3235542993aec5ec0dd` | Lorilee Craker co-authored a memoir about a woman who is what relation to Britney Spears? | `sister` | `mother` | ❌ Incorrect |
| 21 | `5a71532e5542994082a3e7d6` | In what year did the actor who played Edward Snowden act in a movie with Heath Ledger and Julia Styles? | `2007` | `1999` | ❌ Incorrect |
| 22 | `5ae5e5fc55429929b0807a06` | John Shakespeare was the son of the husbandman who lived in what Warwickshire town? | `st john s` | `Snitterfield` | ❌ Incorrect |
| 23 | `5ab985b5554299131ca4235e` | The Crystal Palace Football Club Player of the Year is awarded at the end of each season, nine of these players have won the award for a second time, the most recent being which professional footballer, who plays as a winger for Premier League club Crystal Palace, and the Ivory Coast national team? | `antonio sanchez` | `Wilfried Zaha` | ❌ Incorrect |
| 24 | `5abef2505542993fe9a41db2` | Ulrich Matthes is best known for his role in  a 2004 German-Italian-Austrian historical war drama film directed by who? | `Luc Besson` | `Oliver Hirschbiegel` | ❌ Incorrect |
| 25 | `5a8e0c34554299068b959e37` | The Tremont Stakes in June is among the classic races held at Belmont Park in which U.S. state? | `california` | `New York` | ❌ Incorrect |
| 26 | `5a87dc635542994846c1cdf2` | When was the Polish politician and former Vice-Minister of Foreign Affairs of Poland, who was a Doctor of Law, and who candidated from a right-wing populist, national-conservative, and Christian democratic political party in Poland, elected to the Sejm? | `1 October 1931` | `September 25, 2005` | ❌ Incorrect |
| 27 | `5a81de8d55429926c1cdada9` | Which English actor co-wrote and also played in The Dictator? | `john dillinger` | `Sacha Baron Cohen` | ❌ Incorrect |
| 28 | `5a7559b25542992db9473648` | Nupedia was underwritten by what company that was founded in 1996? | `e-commerce company` | `Bomis` | ❌ Incorrect |
| 29 | `5a8b634255429949d91db58b` | What symbol did a Ghanaian stateswoman, teacher and artist known for designing Ghana's flag in 1957 use in that flag? | `oh i am` | `The Black Star of Africa` | ❌ Incorrect |
| 30 | `5a7f81bc5542992097ad2f8f` | Sanghamitra Bandyopadhyay a professor at theIndian Statistical Institute, a public university established in what year? | `1931` | `1931` | ✅ Correct |
| 31 | `5ab8f1fa5542991b5579f077` | Are Gary Lightbody and Dustin Kensrue from the same country? | `no` | `no` | ✅ Correct |
| 32 | `5a74b6ea55429974ef308c41` | What commune, ecovillage, and spiritual community was founded by Oberto Airaudi? | `saigon` | `Federation of Damanhur` | ❌ Incorrect |
| 33 | `5ab709b15542992aa3b8c78e` | The writers of the "Assassins" screenplay made their directing debut in which year? | `1939` | `1996` | ❌ Incorrect |
| 34 | `5ae7ac1f5542993210983eed` | What is the name of the Los Angeles-based production company that produced a film based on the novel by André Aciman? | `san antonio` | `Water's End Productions` | ❌ Incorrect |
| 35 | `5a74691755429929fddd8412` | What friends star's wedding was photographed by Robert Evans? | `sarah sandler` | `Jennifer Aniston` | ❌ Incorrect |
| 36 | `5a8dea555542995a26add44f` | What is the prequel to the show where Rose Eleanor Arbuthnot-Leslie stars as Maia Rindell? | `The Girl Next Door` | `The Good Wife"` | ❌ Incorrect |
| 37 | `5ab4826f5542990594ba9c3c` | When was Justin Vincent the MVP of a collegiate championship game? | `2002` | `January 4, 2004` | ❌ Incorrect |
| 38 | `5a7735ac5542993735360219` | Which actor starred in Assignment to Kill and passed away in 2000. | `john dillinger` | `John Gielgud` | ❌ Incorrect |
| 39 | `5ab2bfc65542991669774094` | When did the British/Irish digital television channel, which features Spliced, first launch? | `19 January 2007` | `22 July 2002` | ❌ Incorrect |
| 40 | `5ade08425542995b365fabd9` | University of Toronto and Istanbul Technical University, are located in Canada? | `no` | `no` | ✅ Correct |
| 41 | `5ac5093b5542996feb3fe9d8` | Plymouth Regional High School is located in what county? | `Suffolk County` | `Grafton County` | ❌ Incorrect |
| 42 | `5ab26ed65542993be8fa9913` | Does the Inula genus or the Fouquieria genus have more species? | `Inula` | `Inula` | ✅ Correct |
| 43 | `5add1ebe5542990d50227dec` | Who was the director of the 2009 British thriller film starring the actor who is best known for playing Robert Crawley in "Downtown Abbey"? | `John McAdams` | `Anthony Hickox` | ❌ Incorrect |
| 44 | `5a7541ab5542993748c89803` | What school is located in the southernmost atoll of the archipelago of the Maldives? | `sai sai sai sai sai sai sai sai sai sai sai sai s` | `Maradhoo Feydhoo School` | ❌ Incorrect |
| 45 | `5a83e925554299123d8c21ca` | Kofi Dzamesi (born 1960) is a Ghanaian engineer and politician, since February 2017 he has served as Minister of which official Ghanaian agency responsible the creation of linkages between the Government of Ghana and the traditional authorities in the country? | `gubernatorial commission` | `Ministry for Chieftaincy and Traditional Affairs` | ❌ Incorrect |
| 46 | `5a7a783b5542990783324f58` | What country of origin does 17 Carat and Seventeen have in common? | `American` | `Korean` | ❌ Incorrect |
| 47 | `5ae07f6d55429945ae959385` | Who is the protagonist in Ayn Rand's most notable novel? | `samuel samuel` | `Howard Roark` | ❌ Incorrect |
| 48 | `5ab2f57955429929539468cf` | The composer of A Survivor from Warsaw emigrated to the United States in what year? | `1931` | `1934` | ❌ Incorrect |
| 49 | `5a7def2a5542995f4f402324` | Anna Howard Shaw Day starred the actor, director, and producer who was best known for playing what character on Mad Men? | `sam scott` | `Don Draper` | ❌ Incorrect |
| 50 | `5ae2e2c655429928c423952d` | When was the American Democratic politician which was the governor of 185th New York State Legislature born? | `February 17, 1903` | `June 15, 1932` | ❌ Incorrect |
