##################
# NOT USED RIGHT NOW, BUT WILL BE THE BASIS FOR WORKFLOW
##################
from __future__ import annotations
import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Type, Union, get_args, get_origin, get_type_hints

class WorkflowEngine:
    class Event:
        def __init__(self, data: Any = None):
            self.data = data

    class StartEvent(Event):
        pass

    class StopEvent(Event):
        pass

    class AuditedEvent(Event):
        pass

    WorkflowHandler = Callable[[Event, Dict[str, Any]], Awaitable[Any]]
    WorkflowStep = Tuple[
        Tuple[Type[Event], ...],
        Tuple[Type[Event], ...],
        WorkflowHandler
    ]

    def __init__(self):
        self._steps: List[WorkflowEngine.WorkflowStep] = []
        self._queue: asyncio.Queue[WorkflowEngine.Event] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.context: Dict[str, Any] = {}
        self._completion_event = asyncio.Event()

    @classmethod
    def _parse_handler_types(cls, handler: WorkflowEngine.WorkflowHandler) -> Tuple[Tuple[Type[cls.Event], ...], Tuple[Type[cls.Event], ...]]:
        hints = get_type_hints(handler)
        if "event" not in hints:
            raise ValueError(f"Handler '{handler.__name__}' must have an 'event' parameter with an Event type annotation.")

        input_hint = hints["event"]
        if get_origin(input_hint) is Union:
            accepted_events = tuple(
                t for t in get_args(input_hint)
                if t is not type(None) and inspect.isclass(t) and issubclass(t, cls.Event)
            )
        else:
            if not (inspect.isclass(input_hint) and issubclass(input_hint, cls.Event)):
                raise TypeError(f"Handler '{handler.__name__}': 'event' parameter must be a subclass of Event.")
            accepted_events = (input_hint,)

        output_hint = hints.get("return")
        if output_hint is None or output_hint is type(None):
            raise TypeError(f"Handler '{handler.__name__}' must always emit an Event (cannot return None).")

        if get_origin(output_hint) is Union:
            valid_types = tuple(
                t for t in get_args(output_hint)
                if t is not type(None) and inspect.isclass(t) and issubclass(t, cls.Event)
            )
            if not valid_types:
                raise TypeError(f"Handler '{handler.__name__}' must emit at least one valid Event subclass.")
            output_events = valid_types
        else:
            if not (inspect.isclass(output_hint) and issubclass(output_hint, cls.Event)):
                raise TypeError(f"Handler '{handler.__name__}' return type must be a subclass of Event.")
            output_events = (output_hint,)

        final_event = cls.StopEvent
        if any(issubclass(evt, final_event) for evt in accepted_events):
            if not all(issubclass(ot, final_event) for ot in output_events):
                raise TypeError(
                    f"Handler '{handler.__name__}' must emit StopEvent but returns {[ot.__name__ for ot in output_events]}."
                )
        return accepted_events, output_events

    async def register_step(self, handler: WorkflowHandler) -> None:
        accepted_events, output_events = WorkflowEngine._parse_handler_types(handler)
        async with self._lock:
            self._steps.append((accepted_events, output_events, handler))
            input_names = [evt.__name__ for evt in accepted_events]
            output_names = [evt.__name__ for evt in output_events]
            print(f"Registered handler '{handler.__name__}' for input events {input_names} with output options {output_names}")

    def emit(self, event: WorkflowEngine.Event) -> None:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(self._queue.put_nowait, event)
        else:
            asyncio.run(self._queue.put(event))
        print(f"Emitted event '{type(event).__name__}' with data: {event.data}")

    async def _process_event(self, event: WorkflowEngine.Event) -> None:
        handlers_found = False
        new_events = []
        async with self._lock:
            for accepted_events, output_events, handler in self._steps:
                if any(isinstance(event, evt) for evt in accepted_events):
                    handlers_found = True
                    accepted_names = [evt.__name__ for evt in accepted_events]
                    print(f"Dispatching event '{type(event).__name__}' (data: {event.data}) to handler '{handler.__name__}' "
                          f"(accepted types: {accepted_names}) with context: {self.context}")
                    try:
                        result_event = await handler(event, self.context)
                    except Exception as err:
                        print(f"Error in handler '{handler.__name__}': {err}")
                        continue
                    if not isinstance(result_event, WorkflowEngine.Event):
                        raise TypeError(f"Handler '{handler.__name__}' must emit an Event, got {type(result_event).__name__}.")
                    if not any(isinstance(result_event, ot) for ot in output_events):
                        allowed = [ot.__name__ for ot in output_events]
                        raise TypeError(f"Handler '{handler.__name__}' emitted event of type '{type(result_event).__name__}' "
                                        f"which is not in allowed options {allowed}.")
                    new_events.append(result_event)
        if not handlers_found:
            print(f"No registered handler for event '{type(event).__name__}' with data: {event.data}")
        else:
            for new_event in new_events:
                self.emit(new_event)

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


class WorkflowExample:
    class Events:
        class OrderValidatedEvent(WorkflowEngine.Event):
            pass

        class OrderFlaggedEvent(WorkflowEngine.Event):
            pass

        class OrderProcessedEvent(WorkflowEngine.Event):
            pass

    class Handlers:
        @staticmethod
        async def order_validation_handler(event: WorkflowEngine.StartEvent, context: Dict[str, Any]) -> Union[WorkflowExample.Events.OrderValidatedEvent, WorkflowExample.Events.OrderFlaggedEvent]:
            print(f"[order_validation_handler] Received event '{event.__class__.__name__}' with data: {event.data}")
            order = event.data
            order_ids = context.get("order_ids", [])
            order_ids.append(order.get("order_id"))
            context["order_ids"] = order_ids
            print(f"[order_validation_handler] Current order_ids in context: {context['order_ids']}")
            if order.get("total", 0) > 1000:
                context["validation"] = "flagged"
                new_order_data = {**order, "status": "suspicious"}
                print("[order_validation_handler] Order flagged for additional review.")
                return WorkflowExample.Events.OrderFlaggedEvent(new_order_data)
            else:
                context["validation"] = "approved"
                new_order_data = {**order, "status": "validated"}
                print("[order_validation_handler] Order validated successfully.")
                return WorkflowExample.Events.OrderValidatedEvent(new_order_data)

        @staticmethod
        async def inventory_check_handler(event: WorkflowExample.Events.OrderValidatedEvent, context: Dict[str, Any]) -> WorkflowExample.Events.OrderProcessedEvent:
            print(f"[inventory_check_handler] Received event '{event.__class__.__name__}' with data: {event.data}")
            order = event.data
            context["inventory_checked"] = True
            validation_status = context.get("validation")
            print(f"[inventory_check_handler] Validation status from context: {validation_status}")
            await asyncio.sleep(0.5)
            new_order_data = {**order, "inventory": "confirmed"}
            print("[inventory_check_handler] Inventory confirmed for the order.")
            return WorkflowExample.Events.OrderProcessedEvent(new_order_data)

        @staticmethod
        async def fraud_investigation_handler(event: WorkflowExample.Events.OrderFlaggedEvent, context: Dict[str, Any]) -> WorkflowExample.Events.OrderProcessedEvent:
            print(f"[fraud_investigation_handler] Received event '{event.__class__.__name__}' with data: {event.data}")
            order = event.data
            context["fraud_investigated"] = True
            validation_status = context.get("validation")
            print(f"[fraud_investigation_handler] Validation status from context: {validation_status}")
            await asyncio.sleep(0.5)
            new_order_data = {**order, "review": "passed after investigation"}
            print("[fraud_investigation_handler] Fraud investigation completed and order passed review.")
            return WorkflowExample.Events.OrderProcessedEvent(new_order_data)

        @staticmethod
        async def shipping_handler(event: WorkflowExample.Events.OrderProcessedEvent, context: Dict[str, Any]) -> WorkflowEngine.StopEvent:
            print(f"[shipping_handler] Received event '{event.__class__.__name__}' with data: {event.data}")
            order = event.data
            context["shipping_arranged"] = True
            print(f"[shipping_handler] Context before shipping: {context}")
            await asyncio.sleep(0.5)
            new_order_data = {**order, "shipping": "arranged", "finalized": True}
            print("[shipping_handler] Shipping arranged and order processing finalized.")
            context["finalized_order"] = new_order_data
            print(f"[shipping_handler] Final context updated: {context}")
            return WorkflowEngine.StopEvent(new_order_data)

    @staticmethod
    async def main_workflow():
        engine = WorkflowEngine()
        await engine.register_step(WorkflowExample.Handlers.order_validation_handler)
        await engine.register_step(WorkflowExample.Handlers.inventory_check_handler)
        await engine.register_step(WorkflowExample.Handlers.fraud_investigation_handler)
        await engine.register_step(WorkflowExample.Handlers.shipping_handler)


        high_value_order = {"order_id": "ORD456", "total": 1500, "items": ["machine", "toolkit"]}
        print("\n--- Emitting high-value order StartEvent ---")
        engine.emit(WorkflowEngine.StartEvent(high_value_order))

        await engine.run()
        await engine.wait_for_completion()
        print("\nOrder processing workflow complete.")
        print(f"Final context: {engine.context}")

    @staticmethod
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(WorkflowExample.main_workflow())
        finally:
            loop.close()

if __name__ == '__main__':
    WorkflowExample.run()
