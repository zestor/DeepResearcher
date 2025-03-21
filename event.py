import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Type, Union, get_args, get_origin

# Base event class and custom events
class Event:
    def __init__(self, data: Any = None):
        self.data = data

class StartEvent(Event): 
    pass

class IntermediateEvent(Event): 
    pass

# Renamed event: Previously ProcessedEvent, now IntermediateEvent2.
class IntermediateEvent2(Event): 
    pass

class FinalEvent(Event): 
    pass

# Type alias for workflow handler.
WorkflowHandler = Callable[[Any, Dict[str, Any]], Awaitable[Any]]
WorkflowStep = tuple

def _parse_handler_types(handler: WorkflowHandler) -> tuple:
    """
    Extract the accepted input event types and output event type from a handler's type hints.

    Requirements:
      - The 'event' parameter must be annotated with one or more Event subtypes.
      - The return type must be a non-Optional, single Event subtype.
      - If the accepted event type includes FinalEvent, then the output type must be FinalEvent.
    """
    hints = inspect.get_type_hints(handler)
    
    # Validate input event types.
    if "event" not in hints:
        raise ValueError(f"Handler '{handler.__name__}' must have an 'event' parameter with an Event type annotation.")
    
    input_hint = hints["event"]
    if get_origin(input_hint) is Union:
        accepted_events = tuple(
            t for t in get_args(input_hint)
            if t is not type(None) and inspect.isclass(t) and issubclass(t, Event)
        )
    else:
        if not (inspect.isclass(input_hint) and issubclass(input_hint, Event)):
            raise TypeError(f"Handler '{handler.__name__}': 'event' parameter must be a subclass of Event.")
        accepted_events = (input_hint,)
    
    # Validate the return type.
    output_hint = hints.get("return")
    if output_hint is None or output_hint is type(None):
        raise TypeError(f"Handler '{handler.__name__}' must always emit an Event (cannot return None).")

    if get_origin(output_hint) is Union:
        types = get_args(output_hint)
        if type(None) in types:
            raise TypeError(f"Handler '{handler.__name__}' must always emit an Event (None is not allowed).")
        valid_types = [t for t in types if inspect.isclass(t) and issubclass(t, Event)]
        if len(valid_types) != 1:
            raise TypeError(
                f"Handler '{handler.__name__}' return type must be a single Event subclass, got {get_args(output_hint)}."
            )
        output_event = valid_types[0]
    else:
        if not (inspect.isclass(output_hint) and issubclass(output_hint, Event)):
            raise TypeError(f"Handler '{handler.__name__}' return type must be a subclass of Event.")
        output_event = output_hint

    # Enforce that if the handler accepts FinalEvent, its output must be FinalEvent.
    if any(issubclass(evt, FinalEvent) for evt in accepted_events) and not issubclass(output_event, FinalEvent):
        raise TypeError(f"Final handler '{handler.__name__}' must emit FinalEvent but returns {output_event.__name__}.")

    return accepted_events, output_event

class WorkflowEngine:
    def __init__(self):
        self._steps: List[WorkflowStep] = []  # List of (accepted_events, output_event, handler)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.context: Dict[str, Any] = {}
        self._completion_event = asyncio.Event()

    async def register_step(self, handler: WorkflowHandler) -> None:
        """
        Register a workflow step by inferring accepted input and output event types.
        """
        accepted_events, output_event = _parse_handler_types(handler)
        async with self._lock:
            self._steps.append((accepted_events, output_event, handler))
            input_names = [evt.__name__ for evt in accepted_events]
            print(f"Registered handler '{handler.__name__}' for input events {input_names} with output {output_event.__name__}")

    def emit(self, event: Event) -> None:
        """
        Thread-safe event emitter.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(self._queue.put_nowait, event)
        else:
            asyncio.run(self._queue.put(event))
        print(f"Emitted event '{type(event).__name__}' with data: {event.data}")

    async def _process_event(self, event: Event) -> None:
        """
        Dispatch an event to the first handler that accepts its type.
        """
        async with self._lock:
            for accepted_events, _, handler in self._steps:
                if any(isinstance(event, evt) for evt in accepted_events):
                    print(f"Dispatching {type(event).__name__} to handler '{handler.__name__}' with context: {self.context}")
                    try:
                        new_event = await handler(event, self.context)
                    except Exception as err:
                        print(f"Error in handler '{handler.__name__}': {err}")
                        return
                    if not isinstance(new_event, Event):
                        raise TypeError(f"Handler '{handler.__name__}' must emit an Event, got {type(new_event).__name__}.")
                    self.emit(new_event)
                    return
            print(f"No registered handler for event '{type(event).__name__}' with data: {event.data}")

    async def run(self) -> None:
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._queue.empty():
                    self._completion_event.set()
                    break
                continue
            await self._process_event(event)
            self._queue.task_done()

    async def wait_for_completion(self):
        await self._completion_event.wait()

# Workflow handlers

async def step1_handler(event: StartEvent, context: Dict[str, Any]) -> IntermediateEvent:
    print(f"step1_handler received {event.__class__.__name__} with data: {event.data}")
    context["step1"] = "completed"
    await asyncio.sleep(0.5)
    new_data = f"{event.data} -> processed in step1"
    print(f"step1_handler updated context: {context}")
    print("step1_handler emitting IntermediateEvent")
    return IntermediateEvent(new_data)

# Modified step2_handler:
# It emits IntermediateEvent2 (previously ProcessedEvent) that step3_handler will receive.
async def step2_handler(event: IntermediateEvent, context: Dict[str, Any]) -> IntermediateEvent2:
    print(f"step2_handler received {event.__class__.__name__} with data: {event.data}")
    context["step2"] = "completed"
    await asyncio.sleep(0.5)
    previous = context.get("step1", "unknown")
    new_data = f"{event.data} (prev: {previous}) -> processed in step2"
    print(f"step2_handler updated context: {context}")
    print("step2_handler emitting IntermediateEvent2")
    return IntermediateEvent2(new_data)

# Modified step3_handler:
# It now accepts IntermediateEvent2 and returns FinalEvent.
async def step3_handler(event: IntermediateEvent2, context: Dict[str, Any]) -> FinalEvent:
    print(f"step3_handler received {event.__class__.__name__} with data: {event.data}")
    context["workflow_complete"] = True
    await asyncio.sleep(0.5)
    new_data = f"{event.data} -> finalized in step3"
    print(f"step3_handler updated context: {context}")
    print("step3_handler emitting FinalEvent (final event)")
    return FinalEvent(new_data)

async def main_workflow():
    engine = WorkflowEngine()
    await engine.register_step(step1_handler)
    await engine.register_step(step2_handler)
    await engine.register_step(step3_handler)

    engine.emit(StartEvent("Initial Data"))
    await engine.run()
    await engine.wait_for_completion()
    print("Workflow processing complete.")
    print(f"Final context: {engine.context}")

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_workflow())
    finally:
        loop.close()
