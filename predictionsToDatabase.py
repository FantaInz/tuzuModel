import pandas as pd
import numpy as np
import  requests
import psycopg2
import os
import json
def main():
    URL="https://fantasy.premierleague.com/api/event-status/"
    lastUpdate=requests.get(URL).json()["status"][0]["event"]
    df=pd.read_csv("predictions/all_positions_points.csv")
    rows=["ID"]
    for i in range(lastUpdate+1,lastUpdate+6):
        rows.append(f"{i}_Pts")
    points=df[rows]
    conn = psycopg2.connect(database=os.environ["DB_NAME"],
                            user=os.environ["DB_USER"],
                            host=os.environ["DB_HOST"],
                            password=os.environ["DB_PASSWORD"],
                            port=5432)
    playerDf = pd.read_sql("select id,availability from players", conn)
    cur = conn.cursor()
    for row in points.iterrows():

        id=int(row[1].iloc[0])
        availability=playerDf[playerDf["id"]==id].availability
        if availability.empty:
            continue
        mult = 1 if availability.iloc[0] >0 else 0

        arr = ("{" + ", ".join(row[1][1::].apply(lambda x: x*mult).apply(str)) + "}")
        query=f"UPDATE players SET \"expectedPoints\"=\"expectedPoints\"[0:{lastUpdate}]||'{arr}' WHERE id={id}"
        cur.execute(query)
        print("updated expectedPoints for player",id)
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()