# from flask import Flask, jsonify
# from flask_cors import CORS
# from config import Config
# from routes.models import models_bp


# def create_app():
#     app = Flask(__name__)
#     app.config.from_object(Config)

#     # Enable CORS for React frontend
#     CORS(
#         app,
#         origins=["http://localhost:3000"],
#         supports_credentials=True
#     )

#     # ---------------------------
#     # Register blueprints
#     # ---------------------------
#     from routes.health import health_bp
#     from routes.predict import predict_bp
#     from routes.train import train_bp  # train blueprint

#     app.register_blueprint(health_bp, url_prefix="/api/health")
#     app.register_blueprint(predict_bp, url_prefix="/api/predict")
#     app.register_blueprint(train_bp, url_prefix="/api/train")
#     app.register_blueprint(models_bp, url_prefix="/api/models")


#     # ---------------------------
#     # Root route to confirm backend is running
#     # ---------------------------
#     @app.route("/")
#     def root():
#         return jsonify({"status": "ok", "message": "Backend is running!"})

#     return app

# # ---------------------------
# # Run app
# # ---------------------------
# if __name__ == "__main__":
#     app = create_app()
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from routes.models import models_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(
        app,
        origins=["http://localhost:3000"],
        supports_credentials=True
    )

    # Import blueprints
    from routes.health import health_bp
    from routes.predict import predict_bp
    from routes.train import train_bp
    from routes.test import test_bp   # ✅ ADD THIS

    # Register blueprints
    app.register_blueprint(health_bp, url_prefix="/api/health")
    app.register_blueprint(predict_bp, url_prefix="/api/predict")
    app.register_blueprint(train_bp, url_prefix="/api/train")
    app.register_blueprint(test_bp, url_prefix="/api/test")   # ✅ ADD THIS
    app.register_blueprint(models_bp, url_prefix="/api/models")

    @app.route("/")
    def root():
        return jsonify({"status": "ok", "message": "Backend is running!"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
