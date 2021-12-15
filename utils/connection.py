import psycopg2
from utils.config_reader import database_settings
from psycopg2.extras import RealDictCursor
import time
from utils.y_labels import y_columns
from psycopg2 import sql

connector=None
while True:
    try:
        connector=psycopg2.connect(dbname=database_settings.DATABASE,
                                    user=database_settings.DATABASE_USER,
                                    password=database_settings.DATABASE_PASSWORD,
                                    host=database_settings.HOSTNAME,
                                    port=database_settings.PORT,
                                    cursor_factory=RealDictCursor)

        cursor=connector.cursor()
        print("WE MADE A SUCCESSFUL CONNECTION TO THE DATABASE")
        break
    except Exception as error:
        print("CONNECTION FAILED")
        print("ERROR : ",error)
        time.sleep(10)

FEATURE_TABLE=database_settings.TABLES_FEATURES

PREDICTION_TABLE=database_settings.TABLES_PREDICTIONS

CREATE_FEATURES=y_columns

cursor.execute("select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';")
print(cursor.fetchall())

def CREATE_TABLES():
    query=sql.SQL("""
          CREATE TABLE if not exists {table} (
              ID SERIAL PRIMARY KEY,
              QUESTIONS text NOT NULL,
              ANSWERS text NOT NULL,
              TITLE text NOT NULL,
              POSTED_DATE TIMESTAMP Not NULL
          )
          """
    ).format(table=sql.Identifier(FEATURE_TABLE))

    cursor.execute(query)
    # create table to keep the predictions
    query=sql.SQL("""
                  CREATE TABLE if not exists {table} 
                    ( ID SERIAL PRIMARY KEY,
                      PREDICTION_DATE TIMESTAMP Not NULL,
                    {field}
                    )
                """
                ).format(table=sql.Identifier(PREDICTION_TABLE),
                field=sql.SQL(', ').join(
                    [sql.SQL(' ').join([sql.Identifier(col), sql.SQL('decimal'), sql.SQL('Not Null')])
                        for col in CREATE_FEATURES]
                    )
                )

    cursor.execute(query)
    connector.commit()
    print(f"BOTH TABLES {FEATURE_TABLE} AND {PREDICTION_TABLE} ARE CREATED")



if __name__ == '__main__':
    CREATE_TABLES()
    print('loading...')