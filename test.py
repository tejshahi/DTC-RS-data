from tensorflow.keras.models import load_model
from TSClusteringLayer import TSClusteringLayer
from tensorflow.keras.losses import MeanSquaredError

# Load the saved model
loaded_model = load_model('temporal_clustering_model.h5', custom_objects={'TSClusteringLayer': TSClusteringLayer})

# Now you can use the loaded model for inference or further training
print("Model loaded successfully!")

# Check the model’s summary
loaded_model.summary()

# Optionally, check the loss function and optimizer used after loading
print("Loss Function:", loaded_model.loss)

# New data (use real or synthetic time series data here)
n_new_samples = 10
new_data = np.array([ 
    np.stack([ 
        np.sin(x + np.random.rand()), 
        np.cos(x + np.random.rand()), 
        np.sin(2 * x + np.random.rand()) 
    ], axis=1) 
    for _ in range(n_new_samples)
])

# Predict soft assignments (cluster probabilities) for new data
q_new = loaded_model.predict(new_data)[1]

# Assign the cluster with the highest probability
predicted_clusters = np.argmax(q_new, axis=1)

# Print the predicted clusters
print("Predicted clusters for new data:", predicted_clusters)
