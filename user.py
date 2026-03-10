from database import get_connection

class Utilisateur:
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email
        self.history = self.load_history()

    def load_history(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT texte, operation, date_creation FROM historique_textes WHERE user_id=%s ORDER BY date_creation DESC",
            (self.id,)
        )
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
