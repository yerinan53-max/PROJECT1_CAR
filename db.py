import os

import pymysql


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "061153"),
    "database": os.getenv("DB_NAME", "project1_car"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def get_db_connection(include_database=True):
    config = DB_CONFIG.copy()
    if not include_database:
        config.pop("database")
    return pymysql.connect(**config)


def init_database():
    with get_db_connection(include_database=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        connection.commit()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    cylinders FLOAT NOT NULL,
                    displacement FLOAT NOT NULL,
                    horsepower FLOAT NOT NULL,
                    weight FLOAT NOT NULL,
                    acceleration FLOAT NOT NULL,
                    model_year FLOAT NOT NULL,
                    predicted_mpg FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        connection.commit()


def save_prediction(values, predicted_mpg):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    cylinders,
                    displacement,
                    horsepower,
                    weight,
                    acceleration,
                    model_year,
                    predicted_mpg
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    values["cylinders"],
                    values["displacement"],
                    values["horsepower"],
                    values["weight"],
                    values["acceleration"],
                    values["model_year"],
                    predicted_mpg,
                ),
            )
        connection.commit()


def get_recent_predictions(limit=5):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    cylinders,
                    displacement,
                    horsepower,
                    weight,
                    acceleration,
                    model_year,
                    predicted_mpg,
                    DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i') AS created_at
                FROM predictions
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cursor.fetchall()
