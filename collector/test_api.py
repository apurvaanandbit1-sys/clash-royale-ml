from collector.api_client import ClashRoyaleAPI

api = ClashRoyaleAPI()

player = api.get_player("#CG0V8QC9J")

print(player["name"])
print(player["trophies"])
print(player["expLevel"])
print(player["currentDeck"])