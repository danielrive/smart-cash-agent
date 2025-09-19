import logging
import json
from datetime import datetime, timezone, timedelta


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.name,
        }
        # Add extra fields passed to the logger, if any
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        # Include exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def configure_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers if configure_logger() is called multiple times
    if not root.handlers:
        root.addHandler(handler)

#configure_logger()
#logger = logging.getLogger(__name__)



def run():
    from app.agent.agent import graph
    from langchain_core.messages import HumanMessage

    #state = {"messages": [HumanMessage(content="i will buy a present for my mom, i want to register this in my expense app, my local currency is COP but i am buying this in united states, the price is 1000 dollars and this is a bike for my mom")]}
    state = {"messages": [HumanMessage(content="Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.")]}
    #state = {"messages": [HumanMessage(content="I needed to go wiht my pet to the doctor, here the receipt")],"input_file": "./samples/receipt1.jpg" }
    #logger.info("Graph invoked with state: %s", state)
    try:
        result = graph.invoke(state)
        #logger.info("Graph completed, result %s", result)
        print(result["messages"][-1].content)
    except Exception:
        #logger.error("Error in graph execution", exc_info=True)
        raise

if __name__ == "__main__":
    run()
