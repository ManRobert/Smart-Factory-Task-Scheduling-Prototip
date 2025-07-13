
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from deap import base, creator, tools
from collections import defaultdict
import os
import csv

np.random.seed(42)

def load_or_generate_data():
    task_types = ["cut", "assemble", "paint", "test", "package"]
    machines = ["M1", "M2", "M3"]
    history_data = []
    for _ in range(100):
        ttype = random.choice(task_types)
        complexity = random.randint(1, 5)
        machine = random.choice(machines)
        est = random.randint(10, 60)
        real = int(np.clip(np.random.normal(loc=est + complexity * 2, scale=4), 5, 90))
        error = int(np.random.rand() < 0.15)
        history_data.append([ttype, complexity, machine, est, real, error])
    history_df = pd.DataFrame(history_data, columns=["task_type", "complexity", "required_machine", "estimated_duration", "real_duration", "error_occurred"])

    task_data = []
    for i in range(20):
        tid = f"T{i+1}"
        ttype = random.choice(task_types)
        complexity = random.randint(1, 5)
        machine = random.choice(machines)
        est = random.randint(10, 60)
        task_data.append([tid, ttype, complexity, machine, est])
    task_df = pd.DataFrame(task_data, columns=["task_id", "task_type", "complexity", "required_machine", "estimated_duration"])

    return task_df, history_df

def train_models(history_df):
    features = ["task_type", "complexity", "required_machine", "estimated_duration"]
    categorical = ["task_type", "required_machine"]
    numeric = ["complexity", "estimated_duration"]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])
    X = history_df[features]
    y_reg = history_df["real_duration"]
    y_clf = history_df["error_occurred"]
    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    reg_pipeline.fit(X, y_reg)
    clf_pipeline.fit(X, y_clf)
    joblib.dump(reg_pipeline, "model_real_duration.pkl")
    joblib.dump(clf_pipeline, "model_error_risk.pkl")

def optimize_schedule(task_df):
    random.seed(42)
    reg_model = joblib.load("model_real_duration.pkl")
    clf_model = joblib.load("model_error_risk.pkl")
    task_list = list(task_df["task_id"])
    machine_list = ["M1", "M2", "M3"]
    task_features = task_df.set_index("task_id").to_dict("index")
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def create_individual():
        tasks = task_list.copy()
        random.shuffle(tasks)
        allocation = [random.choice(machine_list) for _ in tasks]
        return list(zip(tasks, allocation))

    def repair_individual(individual):
        seen = set()
        repaired = []
        all_tasks = set(task_list)
        for task_id, machine in individual:
            if task_id not in seen:
                repaired.append((task_id, machine))
                seen.add(task_id)
        missing = list(all_tasks - seen)
        for task_id in missing:
            machine = random.choice(machine_list)
            repaired.append((task_id, machine))
        return repaired

    def evaluate(individual):
        total_time = 0
        total_penalty = 0
        for task_id, machine in individual:
            task_info = task_features[task_id]
            input_data = pd.DataFrame([{
                "task_type": task_info["task_type"],
                "complexity": task_info["complexity"],
                "required_machine": machine,
                "estimated_duration": task_info["estimated_duration"]
            }])
            predicted_time = reg_model.predict(input_data)[0]
            proba = clf_model.predict_proba(input_data)[0]
            risk = proba[1] if len(proba) > 1 else 0
            total_time += predicted_time
            total_penalty += 10 * risk
        return (total_time + total_penalty,)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    POP_SIZE = 50
    N_GEN = 40
    MUT_PROB = 0.5
    CX_PROB = 0.5
    STAGNATION_LIMIT = 10
    population = toolbox.population(n=POP_SIZE)
    for ind in population:
        ind[:] = repair_individual(ind)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    best_fitness_per_gen = []
    stagnation_counter = 0
    last_best = None

    for gen in range(N_GEN):
        elite = tools.selBest(population, max(2, int(0.1 * POP_SIZE)))
        offspring = toolbox.select(population, len(population) - len(elite))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                child1[:] = repair_individual(child1)
                child2[:] = repair_individual(child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                mutant[:] = repair_individual(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        population[:] = elite + offspring
        best = tools.selBest(population, 1)[0]
        best_score = best.fitness.values[0]
        best_fitness_per_gen.append(best_score)
        diversity = len(set(tuple(ind) for ind in population)) / len(population)
        print(f"[Generatia {gen + 1}/{N_GEN}]")
        print(f"   Cel mai bun scor: {best_score:.2f}")
        print(f"   Diversitate: {diversity:.2f}")

        if last_best and abs(last_best - best_score) < 0.01:
            stagnation_counter += 1
            if stagnation_counter >= STAGNATION_LIMIT:
                print(">> Stagnare detectata. Oprire timpurie.")
                break
        else:
            stagnation_counter = 0
            last_best = best_score

    with open("fitness_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generatia", "Fitness"])
        for i, score in enumerate(best_fitness_per_gen, 1):
            writer.writerow([i, score])

    plt.plot(best_fitness_per_gen)
    plt.xlabel("Generatia")
    plt.ylabel("Fitness optim")
    plt.title("Evolutia fitnessului")
    plt.grid(True)
    plt.show()
    best_individual = tools.selBest(population, 1)[0]
    print("\nBest Schedule:", best_individual)
    print("Fitness Score:", best_individual.fitness.values[0])
    joblib.dump(best_individual, "best_schedule.pkl")

def show_gantt():
    if not os.path.exists("best_schedule.pkl"):
        print("Fisierul best_schedule.pkl nu exista. Ruleaza intai optimizarea.")
        return

    best_schedule = joblib.load("best_schedule.pkl")
    if not task_df is None:
        task_features = task_df.set_index("task_id").to_dict("index")
    else:
        try:
            task_df_local = pd.read_csv("task_data.csv")
            task_features = task_df_local.set_index("task_id").to_dict("index")
        except:
            print("Nu s-au gasit taskurile. Ruleaza intai optiunea 1.")
            return

    reg_model = joblib.load("model_real_duration.pkl")
    data = []
    current_time_per_machine = {"M1": 0, "M2": 0, "M3": 0}
    colors = {"M1": "tab:blue", "M2": "tab:green", "M3": "tab:red"}

    for task_id, machine in best_schedule:
        info = task_features[task_id]
        input_df = pd.DataFrame([{
            "task_type": info["task_type"],
            "complexity": info["complexity"],
            "required_machine": machine,
            "estimated_duration": info["estimated_duration"]
        }])
        predicted = reg_model.predict(input_df)[0]
        start = current_time_per_machine[machine]
        end = start + predicted
        current_time_per_machine[machine] = end
        data.append((task_id, machine, start, end))

    fig, ax = plt.subplots()
    for task_id, machine, start, end in data:
        ax.barh(machine, end - start, left=start, color=colors[machine], edgecolor='black')
        ax.text(start + (end - start)/2, machine, task_id, va='center', ha='center', color='white', fontsize=8)

    ax.set_xlabel("Timp")
    ax.set_ylabel("Masina")
    ax.set_title("Diagrama Gantt - Programare Taskuri")
    plt.tight_layout()
    plt.show()

def meniu():
    global task_df, history_df
    task_df = None
    history_df = None
    while True:
        print("\n==== MENIU SMART FACTORY OPTIMIZAT ====")
        print("1. Generare date")
        print("2. Antrenare modele")
        print("3. Optimizare cu GA")
        print("4. Afisare Gantt")
        print("5. Iesire")

        opt = input("Selecteaza o optiune: ").strip()
        if opt == "1":
            task_df, history_df = load_or_generate_data()
            print("\n>> Date istorice pentru model (primele 5 intrări):")
            print(history_df.head())
            print("\n>> Taskuri curente de programat (primele 5 intrări):")
            print(task_df.head())
            print(f"\n>> Total taskuri generate: {len(task_df)}")
            print(">> Total înregistrări istorice: ", len(history_df))
        elif opt == "2":
            if history_df is None:
                print("Genereaza intai date.")
            else:
                train_models(history_df)
                print("\n>> Modelele RandomForest au fost antrenate cu succes.")
                print("   - model_real_duration.pkl (regresie pentru durată)")
                print("   - model_error_risk.pkl (clasificare pentru risc)")
        elif opt == "3":
            if task_df is None or not os.path.exists("model_real_duration.pkl"):
                print("Genereaza date si antreneaza modelele inainte.")
            else:
                optimize_schedule(task_df)
        elif opt == "5":
            break
        elif opt == "4":
            show_gantt()
        else:
            print("Optiune invalida.")



if __name__ == "__main__":
    meniu()
