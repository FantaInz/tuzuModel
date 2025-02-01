# **Tuzu Model**  

A set of scripts to create **point predictions** for Fantasy Premier League, based on the great repository:  
[**vaastav/Fantasy-Premier-League**](https://github.com/vaastav/Fantasy-Premier-League).  

To ensure **accurate predictions**, the models are trained using only matches where players played at least **60 minutes**. Then future predictions are generated based on model results, xMins and availability.  

I plan to improve the model in the future, and I **love discussing FPL and FPL prediction models**.  
Feel free to reach out to me at **szymone25@gmail.com**.  

A **frontend app** for this project is available here:  
[**minifpl-front**](https://github.com/FantaInz/minifpl-front)  
You are welcome to both of these and adapt them to your needs.  

---

## **How to use** 

### **Run the notebooks manually**

1. Clone the repository and navigate to the project folder:  
   ```bash
   git clone https://github.com/FantaInz/tuzuModel.git
   cd tuzuModel
   ```

2. Set up a virtual environment:  
   ```bash
   bash setup_venv.sh <name_of_env>
   source <name_of_env>/bin/activate
   ```

3. Start Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
### **Run everything automatically**

1. Clone the repository and navigate to the project folder:  
   ```bash
   git clone https://github.com/FantaInz/tuzuModel.git
   cd tuzuModel
   ```

2. Set up a virtual environment:  
   ```bash
   bash setup_venv.sh <name_of_env>
   source <name_of_env>/bin/activate
   ```
   
3. Run the entire workflow (with optional database saving):  
   If you want to save predictions to the database, set the appropriate database variables before running the script:  
   ```bash
   export DB_NAME=my_database
   export DB_USER=my_user
   export DB_HOST=my_host
   export DB_PASSWORD=my_password
   export PORT=your_port
   ```

   Then run the script:  
   ```bash
   RUN_SAVE_TO_DB=true bash execute_whole.sh  # Set to true if saving to DB, otherwise omit RUN_SAVE_TO_DB
   ```

### **Run in Docker**

1. Download and install Docker:  
   [Docker Installation Guide](https://docs.docker.com/desktop/)

2. Clone the repository and navigate to the project folder:  
   ```bash
   git clone https://github.com/FantaInz/tuzuModel.git
   cd tuzuModel
   ```

3. Build the Docker image:  
   ```bash
   docker build -t <app_name> .
   ```

4. Run the container:  

   - **Without saving predictions to the database:**  
     ```bash
     docker run --rm <app_name>
     ```

   - **With saving predictions to the database:**  
     Make sure to set the correct database environment variables:  
     ```bash
     docker run --rm \
         -e DB_NAME=my_database \
         -e DB_USER=my_user \
         -e DB_HOST=my_host \
         -e DB_PASSWORD=my_password \
         -e PORT=your_port \
         -e RUN_SAVE_TO_DB=true \
         <app_name>
     ```

---

## **Folder Structure**  

- **`predictions/`** – Contains predictions for upcoming gameweeks:  
  - **`-raw`** → Predictions without xMins adjustments.  
  - **`no-availability`** → Predictions with xMins but without considering player availability.  

- **`models/`** – Contains trained models.  

- **`created_csv/`** – Contains mappings between **FPL** and **Understat** data, as well as player names across seasons.  

---

## **Flow**  

| **Step** | **Notebook** | **Description** |
|----------|-------------|----------------|
| **1** | `repo_cleanup.ipynb` | Retrieves data and deletes unnecessary files and folders. |
| **2** | `entry_processing.ipynb` | Preprocesses the data to prepare it for training and predictions. |
| **3** | `gk_training.ipynb` | Trains a **CatBoost model** for **goalkeepers**. |
| **4** | `def_training.ipynb` | Trains a **Random Forest model** for **defenders**. |
| **5** | `mid_training.ipynb` | Trains a **CatBoost model** for **midfielders**. |
| **6** | `fwd_training.ipynb` | Trains a **CatBoost model** for **forwards**. |
| **7** | `get_prediction_data.ipynb` | Calculates the data needed for predicting the next gameweeks. |
| **8** | `predictions.ipynb` | Generates predictions **with and without xMins adjustments**. |
| **9** | `add_availability.ipynb` | Fetches **player availability** from the **FPL API** and adjusts predictions. |
| **10** | `save_to_database.ipynb` | Saves predictions to the database (used in [**BackEnd**](https://github.com/FantaInz/BackEnd)). |

