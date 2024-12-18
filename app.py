from flask import Flask, request, jsonify
from sklearn.decomposition import PCA
import numpy as np

app = Flask(__name__)

@app.route('/pca', methods=['POST'])
def process_pca():
    data = request.json
    embeddings = np.array(data['embedding'])
    
    print("Received Embeddings Shape:", embeddings.shape)  # Log shape
    
    if embeddings.ndim != 2:
        return jsonify({ 'error': 'Expected a 2D array of embeddings.' }), 400

    if embeddings.shape[0] < 2:
        return jsonify({ 'error': 'PCA requires multiple samples (embeddings)' }), 400
    # Ensure n_components is smaller than the number of features
    n_components = min(embeddings.shape[0], embeddings.shape[1])  # Adjust n_components

    #n_components = min(50, embeddings.shape[1])  # Adjust to desired number of components

    # Perform PCA to reduce the embedding dimensions
        # Perform PCA to reduce the embedding dimensions
    pca = PCA(n_components=n_components, svd_solver='full')
    reduced_embeddings = pca.fit_transform(embeddings)
    
    return jsonify({ 'reduced_embedding': reduced_embeddings.tolist() })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5030, debug=True )