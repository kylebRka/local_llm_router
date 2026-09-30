"""Generate 1,500 English examples: 500 per model tier, five forms per topic."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data' / 'training' / 'requests_en.jsonl'

SIMPLE = {
 'general': ['computer file','web browser','email address','password','backup copy','calendar event','shopping list','bus timetable','weather forecast','library card','coffee recipe','boiled egg','rice recipe','sandwich','garden plant','washing machine','light bulb','phone charger','train ticket','dictionary'],
 'mathematics': ['fraction','percentage','average','median','odd number','even number','triangle','rectangle','right angle','number line','decimal number','negative number','multiplication','division','square root','rounding','time interval','unit conversion','ratio','simple probability'],
 'science': ['photosynthesis','evaporation','condensation','rainbow','gravity','magnetism','plant roots','water cycle','day and night','seasons','sound wave','reflection of light','melting ice','freezing water','air pressure','clouds','seed germination','food chain','solar energy','recycling'],
 'programming': ['variable in Python','for loop','list in Python','string in Python','if statement','function in programming','integer type','Boolean value','dictionary in Python','return statement','comment in code','command line','folder path','file extension','web page','HTML tag','CSS rule','JSON object','API request','software bug'],
 'writing': ['a short birthday greeting','a polite thank-you note','a meeting reminder','a simple invitation','a one-line apology','a subject line for a school email','a friendly welcome message','a short event title','a postcard greeting','a brief congratulations','a caption for a photo','a two-line announcement','a simple to-do list','a polite request for a receipt','a short farewell note','a reminder to bring a book','a note to a neighbor','a simple holiday wish','a brief status update','a one-sentence summary'],
}
MEDIUM_ACTIONS = [
 'write a Python function to count words in', 'write a Python script to deduplicate',
 'write a Python function to sort', 'write a Python function to validate',
 'write a Python script to group', 'write a Python function to filter',
 'write a Python script to summarize', 'write a Python function to find duplicates in',
 'write a Python script to convert', 'write a Python function to search',
 'write a Python script to merge', 'write a Python function to normalize',
 'write a Python script to calculate totals from', 'write a Python function to split',
 'write a Python script to parse', 'write a Python function to format',
 'write a Python script to compare', 'write a Python function to sample',
 'write a Python script to remove empty rows from', 'write a Python function to rank',
]
MEDIUM_OBJECTS = ['a CSV of shop orders','a list of school grades','a JSON file of book titles','a table of monthly expenses','a log of website visits']
HARD_SYSTEMS = [
 'a multi-region payment platform','a distributed inventory service','a real-time fraud detection pipeline',
 'a large-scale search index','a global chat service','a durable event streaming platform',
 'a recommendation system for millions of users','a hospital data exchange','a financial audit pipeline',
 'a fault-tolerant job scheduler','a multi-tenant analytics platform','a global media delivery system',
 'a high-volume reservation system','a privacy-preserving identity service','a distributed key-value store',
 'a cross-region backup system','a real-time telemetry platform','a large online marketplace',
 'a resilient API gateway','a supply chain tracking network',
]
HARD_CONSTRAINTS = [
 'during a regional outage', 'under strict data residency rules',
 'with zero-downtime schema changes', 'under sudden traffic spikes',
 'while preserving exactly-once business effects',
]


def build_records():
    rows=[]
    for category, subjects in SIMPLE.items():
        for subject in subjects:
            family=f'en:e2b:{category}:{subject}'
            if category == 'writing':
                forms=[f'Write {subject}.',f'Draft {subject} in plain English.',f'Give me {subject} in one sentence.',f'Create {subject} with a friendly tone.',f'Please make {subject} concise.']
            else:
                forms=[f'What is {subject}?',f'Explain {subject} in simple terms.',f'Give a short definition of {subject}.',f'What does {subject} mean to a beginner?',f'Give one easy example of {subject}.']
            rows.extend({'text':x,'label':'E2B','category':category,'difficulty':1,'family':family,'language':'en'} for x in forms)
    # 40 programming families and 60 families of other medium tasks.
    for action in MEDIUM_ACTIONS[:8]:
        for obj in MEDIUM_OBJECTS:
            family=f'en:e4b:{action}:{obj}'
            task=f'{action} {obj}'
            forms=[f'{task.capitalize()}.',f'Please {task}, with a small example.',f'Can you {task} and explain the main steps?',f'{task.capitalize()}; include basic error handling.',f'Give working code to {task}, plus one test case.']
            rows.extend({'text':x,'label':'E4B','category':'programming','difficulty':4,'family':family,'language':'en'} for x in forms)
    medium_topics = {
      'mathematics': ['a two-step percentage discount','an average from grouped scores','a linear equation with fractions','a train arrival time across time zones','a recipe scaled for twelve people','a compound interest estimate','a weighted grade average','an area from two connected rectangles','a probability with two dice','a unit conversion table','a median after removing an outlier','a budget split among four categories','a travel speed from distance and time','a ratio from a word problem','a sequence with a missing term','a quadratic equation by factoring','a monthly loan payment estimate','a fuel cost across three routes','a percentage change over two years','a simple system of two equations'],
      'writing': ['a professional email requesting a deadline extension','a concise cover letter for a support role','a two-paragraph product description','a meeting summary with action items','a customer support reply about a delayed order','a short presentation outline about recycling','a newsletter introduction for a local library','a polite complaint about a billing error','an announcement of changed office hours','a short speech for a school event','a FAQ for a small online store','a one-page project proposal','a welcome email for new volunteers','a comparison of two book summaries','a clear rewrite of a technical paragraph','a step-by-step user guide for a basic app','a short social media campaign plan','a three-part lesson outline','a balanced review of a restaurant','a clear status report for a team'],
      'analysis': ['two commuting options by time and cost','two phone plans by monthly usage','three study schedules for a student','a small store sales trend','customer feedback on a new menu','a simple A/B test result','pros and cons of remote work for a small team','two approaches to reduce food waste','a weekly exercise plan under time limits','a choice between renting and buying equipment','risks of launching a small newsletter','a plan to improve library attendance','a basic survey design for a café','a comparison of two energy saving options','a small project timeline','a customer churn summary','a travel itinerary under a budget','a product feature prioritization list','a team meeting format','a short experiment to test a hypothesis'],
    }
    medium_forms = {
      'mathematics': ['Solve {topic} and show the steps.','Work through {topic} with a clear calculation.','Explain how to calculate {topic}, then give the result.','Check the arithmetic for {topic} and show your method.','Give a worked example of {topic} with intermediate steps.'],
      'writing': ['Draft {topic} with a clear structure.','Write {topic} for a general audience.','Create {topic} and explain the tone choices briefly.','Prepare {topic} in a professional style.','Give a polished version of {topic} with concise wording.'],
      'analysis': ['Compare {topic} using clear criteria.','Analyze {topic} and recommend a practical choice.','Evaluate {topic} with benefits and drawbacks.','Outline a decision framework for {topic}.','Summarize the evidence around {topic} and identify trade-offs.'],
    }
    for category, topics in medium_topics.items():
        for topic in topics:
            family=f'en:e4b:{category}:{topic}'
            rows.extend({'text':form.format(topic=topic),'label':'E4B','category':category,'difficulty':4,'family':family,'language':'en'} for form in medium_forms[category])
    # 40 architecture families and 60 complex programming, math and analysis families.
    for system in HARD_SYSTEMS[:8]:
        for constraint in HARD_CONSTRAINTS:
            family=f'en:12b:{system}:{constraint}'
            task=f'design {system} {constraint}'
            forms=[f'Architect {system} {constraint}; explain trade-offs.',f'How would you {task}? Cover failure modes.',f'Create a detailed architecture for {system} {constraint}, including recovery.',f'Evaluate design options for {system} {constraint} and justify one.',f'Plan {system} {constraint}; specify consistency, monitoring and rollback.']
            rows.extend({'text':x,'label':'12B','category':'systems','difficulty':8,'family':family,'language':'en'} for x in forms)
    hard_topics = {
      'programming': ['a concurrency bug in an event-driven service','a memory leak across worker processes','a migration from a monolith to services','a safe retry design for payments','a distributed cache invalidation policy','a zero-downtime database migration','a secure multi-tenant authorization layer','a streaming data backfill','a dependency upgrade across many services','a fault-tolerant message consumer','a performance regression in a large codebase','a cross-region failover controller','a reproducible build pipeline','a transactional outbox implementation','a rate limiter across many nodes','a privacy-safe logging architecture','a scheduler with cancellation and retries','a complex API versioning strategy','a large-scale search ranking pipeline','a secure secret rotation mechanism'],
      'mathematics': ['a convergence proof for an iterative method','an error bound for numerical integration','a stability analysis for a differential equation','a Bayesian model with correlated evidence','a constrained optimization with nonlinear costs','a proof of correctness for a graph algorithm','a queueing model with bursty arrivals','a stochastic process with absorbing states','a sensitivity analysis for a multivariate model','a formal derivation of a recurrence relation','an asymptotic complexity proof','a Monte Carlo variance reduction method','a robust regression under heavy outliers','a scheduling optimization with constraints','a matrix decomposition under rank deficiency','a probability model for dependent failures','a numerical solver for a stiff system','a multi-objective optimization trade-off','a hypothesis test with multiple comparisons','a geometric proof involving several lemmas'],
      'analysis': ['a causal impact study with confounding factors','a multi-year product strategy under uncertain demand','a regional public health intervention comparison','a risk analysis for a payment platform launch','a policy decision with conflicting stakeholder goals','an experiment affected by selection bias','a forecast with regime changes','a privacy and utility trade-off in analytics','a supply chain resilience plan','an incident review across multiple services','a market entry strategy under regulation','a longitudinal study with missing data','a resource allocation plan across hospitals','a migration strategy for a large organization','an evaluation of competing climate scenarios','a governance plan for sensitive data','a high-stakes decision under incomplete evidence','a portfolio of interacting operational risks','a trust and safety strategy at scale','an audit of a complex recommendation system'],
    }
    hard_forms = {
      'programming': ['Design and justify a production solution for {topic}; cover failure recovery.','Analyze {topic} and propose a robust implementation with tests.','Plan how to solve {topic} under load and explain trade-offs.','Investigate {topic}, including edge cases and rollback.','Give a detailed engineering approach to {topic}, with monitoring and validation.'],
      'mathematics': ['Derive {topic} rigorously and state assumptions.','Work through {topic} with a formal argument and limitations.','Prove or develop {topic} step by step.','Analyze {topic}, including boundary cases.','Give a rigorous solution for {topic} and verify the result.'],
      'analysis': ['Develop a detailed approach to {topic} and discuss uncertainty.','Evaluate {topic} with competing explanations and limitations.','Create a rigorous plan for {topic}, including validation.','Compare strategies for {topic} and identify failure modes.','Analyze {topic} deeply and defend a recommendation.'],
    }
    for category, topics in hard_topics.items():
        for topic in topics:
            family=f'en:12b:{category}:{topic}'
            rows.extend({'text':form.format(topic=topic),'label':'12B','category':category,'difficulty':8,'family':family,'language':'en'} for form in hard_forms[category])
    if Counter(row['label'] for row in rows) != {'E2B':500,'E4B':500,'12B':500}:
        raise ValueError('Expected 500 examples per tier')
    if len({row['text'].casefold() for row in rows}) != len(rows):
        raise ValueError('Duplicate English examples')
    return rows


def main():
    rows=build_records()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    print(f'Saved / Сохранено {len(rows)} English examples / английских запросов: {OUTPUT}')


if __name__=='__main__':
    main()
