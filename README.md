**Oncolytic Virotherapy Optimisation**

This project models and optimises tumour treatment schedules using oncolytic viruses, inspired by Beata Halassy’s breast cancer treatment. The code implements mathematical models, grid search, and genetic algorithms to refine timing and dosage for effective treatment outcomes.

**Features**

Tumour Growth Modelling: Uses a logistic growth model with exponential decay for tumour volume prediction.

Grid Search Optimisation: Identifies optimal timing for MeV and VSV injections.

Genetic Algorithm: Refines dosage combinations for maximum tumour reduction.

Visualisation: Graphs tumour size progression and dosage evolution over time.

**How to Run**
1. Install dependencies:

pip install numpy scipy matplotlib

2. Run the tumour growth and decay model:

python tumor_model.py

3. Run the grid search for optimal timing:
   
python grid_search.py

4. Run the genetic algorithm for dosage optimisation:

python genetic_algorithm.py

**Results**

- Optimal timing and dosage strategies are printed in the console.
- Visualisations display tumour size progression and dosage evolution.

**Notes**

Parameters like tumour size, viral decay rates, and dosage ranges can be modified within the code for further experimentation.

The project serves as a starting point for refining oncolytic virotherapy model
