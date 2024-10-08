import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

#connect to the db

con = psycopg2.connect(
    host = "Wills-Computer.local",
    database = "fgsales",
    user = "postgres",
    password = "postgres")




#execute query to select date and time

query = ("SELECT date, total_day from Fgsales2024_q1")

#use pandas to execute query and store it in dataframe

df = pd.read_sql(query, con)

#close connection
con.close()
5
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["total_day"], marker="o")
plt.title("Daily total Sales (High to low)")
plt.xlabel("Date")
plt.ylabel("Total Sales")
#plt.xticks(rolation=45) #rotate data for better visibility
plt.grid(True)

#Show the plot

plt.tight_layout()
plt.show()





query.index()






#close cursor
cur.close()
#close the connection
  
con.close()

