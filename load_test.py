
from locust import HttpUser, task, between

# Este token es solo un ejemplo; reemplázalo con tu token real
token = ""

class MyApiUser(HttpUser):
    # Espera aleatoria entre peticiones: de 1 a 3 segundos
    wait_time = between(1, 3)

    def on_start(self):
        # Asigna el token de autenticación
        self.token = token  # <--- Reemplaza con tu token real

        # Headers que se usarán en las peticiones
        self.headers = {
            "Authorization": self.token,
            "Content-Type": "application/json"
        }

    @task
    def call_api(self):
        # Ejemplo de petición GET
        self.client.get(
            "/docs",  # Ajusta a tu ruta real (por ejemplo '/', '/predict', etc.)
            headers=self.headers
        )

        # Ejemplo de petición POST (descomentarlo si lo necesitas)
        self.client.post(
             "/prediccion",
             json={
                  "Gender": "1",
                  "Age": 27,
                  "Height": 1.65,
                  "Weight": 70,
                  "family_history_with_overweight": "1",
                  "FAVC": "1",
                  "FCVC": 3,
                  "NCP": 3,
                  "CAEC": "1",
                  "SMOKE": "0",
                  "CH2O": 2,
                  "SCC": "1",
                  "FAF": 0,
                  "TUE": 0,
                  "CALC": "2",
                  "MTRANS": "1"
                },
             headers=self.headers
        )




