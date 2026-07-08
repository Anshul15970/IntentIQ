from utils.intent_mapper import IntentMapper

mapper = IntentMapper()

prediction = "card_delivery_status"

mapped = mapper.map_prediction(prediction)

print("Original :", prediction)
print("Mapped   :", mapped)