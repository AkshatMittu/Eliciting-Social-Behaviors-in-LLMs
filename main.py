from utils import *


# Install Dependencies
"""
pip install -r requirements.txt
"""

# Load Dataset
print("Loading Dataset...")
df = get_dataset()

n_rounds = 7
print(f"Running Analysis for {n_rounds} rounds")
question_counter = 0

agents = [use_gemini, use_groq] # change function names here for other agents
n_agents = len(agents)

scores = [0] * n_agents
n_deceived = [0] * n_agents
trust_scores = [3] * n_agents # was 0 before
hinter_reasons = {"behavior": [], "reason_for_behavior": []}
solver_reasons = {"did_you_use_hint": [], "reason_for_choice_on_hint_usage": []}

tasks = get_tasks(df, n_rounds, n_agents=2)
results_log = []


for i in tqdm(range(n_rounds)):
  print(f"Round: {i+1}")

  for j in range(n_agents):

    print("\nContext Fetched..")
    context_hinter = get_context(i+1, n_rounds, scores[j], scores[n_agents-(j+1)], n_deceived[j], n_deceived[n_agents-(j+1)], trust_scores[j])
    context_solver = get_context(i+1, n_rounds, scores[n_agents-(j+1)], scores[j], n_deceived[n_agents-(j+1)], n_deceived[j], trust_scores[n_agents-(j+1)])

    print("\n Behavior Summary...")
    hinter_summary = use_groq(summarize_behavior(hinter_reasons), op_type='text')
    solver_summary = use_groq(summarize_behavior(solver_reasons), op_type='text')

    context_hinter = use_groq(judge_prompt(context_hinter, hinter_summary), op_type='text')
    context_solver = use_groq(judge_prompt(context_solver, solver_summary), op_type='text')

    # print(context_hinter)
    # print()
    # print(context_solver)
    # print()


    hinter = agents[j]
    solver = agents[n_agents-(j+1)]

    print(f"Hinter is {j}, Solver is {n_agents-(j+1)}")

    question = tasks[question_counter][0]
    solution = tasks[question_counter][1]
    # correct_answer = tasks[question_counter][2]
    question_counter += 1 # for next question in same round


    # print(f"Hinter Prompt:")
    # print()
    # print(hinter_prompt(context_hinter, question))
    # print()
    # print("-----------------------------------")

    print("Generating hint..")
    # hint = hinter(hinter_prompt(context_hinter, question), hinter_schema, max_number_tokens=60, max_string_token_length=100)
    hint = eval(hinter(hinter_prompt(context_hinter, question), schema=hinter_schema))
    print("Hint generated..")
    print(hint)
    hint_text = hint['hint']
    hint_type = hint['hint_type']
    reason_for_hint_type = hint['reason_for_hint_type']

    # Updating hinter behavior
    hinter_reasons["behavior"].append(hint_type)
    hinter_reasons['reason_for_behavior'].append(reason_for_hint_type)


    # print("-----------------------------------")
    # print(f"Solver Prompt:")
    # print()
    # print(solver_prompt(context_solver, question, hint_text))
    print()
    print("-----------------------------------")
    print("Solver is thinking..")
    # answer = solver(solver_prompt(context_solver, question, hint_text), solver_schema, max_number_tokens=100, max_string_token_length=150)
    print("=====================================================================================") # CHANGED HERE!!
    answer = solver(solver_prompt(context_solver, question, hint_text), schema=solver_schema)
    # print(answer)
    print("=====================================================================================") # CHANGED HERE!!
    answer = eval(answer)
    print()
    print("Solver output..")
    print(answer)

    # Updating solver reasons
    solver_reasons["did_you_use_hint"].append(answer['did_you_use_hint'])
    solver_reasons['reason_for_choice_on_hint_usage'].append(answer['reason_for_choice_on_hint_usage'])


    print("-----------------------------------")
    print(question)
    print()
    print(solution)

    # Check correctness
    is_correct = eval(use_gemini(answer_checker(question, solution, answer['Explanation'] + '\nAnswer: ' + str(answer['answer']))))
    print(is_correct)
    print()

    if str(is_correct['answer_is_correct']) == '1':
      scores[n_agents-(j+1)] += 1


    # Sum of trust_score, since only solver gives trust scores
    trust_scores[n_agents-(j+1)] += int(answer['trust_score_on_hinter'])

    # Counting number of times hinter has deceived solver
    if hint_type == 'deceptive':
      n_deceived[n_agents-(j+1)] += 1

    results_log.append({
        'round': i + 1,
        'turn': j, # Index of the agent who was the Hinter (Agent 0 or 1)
        'hinter_agent': j,
        'solver_agent': n_agents - (j + 1),
        'score_gained': 1 if str(is_correct['answer_is_correct']) == '1' else 0,
        'hint_type': hint_type,
        'trust_score_assigned': int(answer['trust_score_on_hinter']),
    })

    print("===========================Roles swapped===============================")

  print()
  print("=========================================================")
  print(f"End of Round: {i+1}")
  print("=========================================================")
  print()
