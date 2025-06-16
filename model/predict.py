import pickle
import pandas as pd

# Load the model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# come from MLFLOW
MODEL_VERSION = '1.0.0'  

# get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()  

def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    #predict the class
    predicted_class = model.predict(df)[0]

    # Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    # create mapping: {class_name: probability}
    class_probs = dict(zip(class_labels,map(lambda p:round(p,4), probabilities)))

    return {
        "predited_Category": predicted_class,
        "confidence":round(confidence,4),
        "class_probabilities": class_probs
    }