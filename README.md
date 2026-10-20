# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution

### Task 1

My verifier is a prompt chain with deterministic tool use rather than a free
tool-calling loop, following the "let the LLM read, let code compute" idea from
Tutorial 1. (1) **Extract:** `deepseek-v4-flash` (thinking disabled,
temperature 0) turns the CV text into JSON claims (name, current city, jobs
with company/title/years, education, skills); the prompt treats the CV as data
and tolerates text split across PDF table cells. (2) **Retrieve:** Python calls
`search_linkedin_people` with the name plus city, country and the industry from
the headline, fetches up to 12 candidate profiles with `get_linkedin_profile`,
and picks the one that best matches the CV (same name, city, companies, start
years, schools, skills), because many candidates share a name. (3) **Compare:**
every claim is checked field by field in code: name, city, each company must
exist with the same start/end years and "Present" status, seniority words
(Senior/Junior/Mid) must match the LinkedIn seniority, degree levels are mapped
to codes (BSc, MSc, MBA, PhD, ...), graduation years must match, and each skill
must be on the profile. (4) **Judge:** only wording questions that code cannot
settle (e.g. "Stanford" vs "Stanford University", "ML" vs "Machine Learning")
are sent to the LLM as short pairs, never the whole CV. (5) **Score:** any
discrepancy gives 0.1, none gives 0.9, no profile found gives 0.2. A guard gives
0.1 to CVs with hidden instructions, zero-width characters or homoglyphs. CVs
run 3 at a time (`asyncio.Semaphore(3)`), with retries and a per-CV timeout so
one failure cannot crash the run.

Results on `public_test`: **7/7 correct (CV_4: start year 2015 vs 2014; CV_5: Senior Analyst vs junior Analyst; CV_7: Docker not on LinkedIn), about 1 minute per run.**

### Task 2

**Technique: split the visible layer from the text layer of the PDF**
(hidden text / PDF text extraction). `task2/make_adversarial_cv.py` builds
`task2/adversarial_cv.pdf` from two layers:

* What a human sees is a **picture** (PNG) of an embellished CV. Images contain
  no text, so the PDF loader used by the verifiers (`markitdown`/`pdfminer`)
  extracts nothing from it.
* On top of it is Kelly Tsang's **true** CV written in PDF text render mode 3
  ("invisible text", the mode OCR tools use for scanned pages). It is never
  painted, but every text extractor reads it.

The false details, visible only to a human reader: the degree is upgraded to
**MSc in Finance** (LinkedIn: BSc), the AIA job starts in **2021** (LinkedIn:
2022), and the title is inflated to **Senior Manager** (LinkedIn: Manager).

Why it should work: the verifiers only see `cv_text()` output, and for this PDF
that output is word for word identical to the true CV's text, which agrees with
her LinkedIn and Facebook profiles. So the false claims never reach the LLM and
there is nothing for any verifier to catch, however well it checks the fields.
It also contains no injected instructions, Unicode tricks or unusual words, so
defences that look for prompt injection (like the guard in my own Task 1
agent) have nothing to flag. My own agent scores it the same as the true CV.
The lesson for KYC: an agent should verify what the human-facing document
shows (e.g. render the page and OCR it, or compare the text layer with an OCR
of the rendering), not only the extracted text.
