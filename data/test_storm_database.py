from StormDatabase import StormDatabase

db = StormDatabase()

print("Total entries:", db.totalEntries())
print("Storm count:", db.stormCount())
print("First 10 IDs:", db.getStormIds()[:10])
print("AL011851 records: \n", db.getStormRecord("AL011851"))