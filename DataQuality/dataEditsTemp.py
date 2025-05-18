import pandas as pd

df = pd.read_csv("Data/Dairycomp_data/Archive_EVENTS_2S ID CBRD BDAT REG SID SREG DID PEN_4_22_2025_EVENTS.CSV", sep=",", on_bad_lines="error")
df2 = pd.read_csv("Data/Dairycomp_data/current_EVENTS_2S ID CBRD BDAT REG SID SREG DID PEN_4_22_2025_EVENTS.CSV", sep=",", on_bad_lines="error")
df3 = pd.concat([df, df2])
df3["ID"] = pd.to_numeric(df["ID"], errors="coerce")
df3["BDAT"] = pd.to_datetime(df["BDAT"], errors="coerce")
df3["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#Removes duplicate lines
df3.drop_duplicates(inplace=True)
#Sorts by ID, then by Birthday
df3.sort_values(by=["ID", "BDAT"], ascending=[True, True], inplace=True)

#Groups by ID number, then drops the older cows, keeping only the newest for that ID
df3["max_BDAT"] = df3.groupby("ID")["BDAT"].transform("max")
df3 = df3[df3["BDAT"] == df3["max_BDAT"]].drop(columns=["max_BDAT"])

df3.to_csv("Data/Dairycomp_data/mergedData", sep=",")

print(len(df3))