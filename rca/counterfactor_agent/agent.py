from .controller import CounterfactualController
from .executor import CounterfactualExecutor


class CounterfactualRCAAgent:
    """Thin orchestrator: wires a fresh Executor + Controller pair per call.

    - Executor (executor.py): talks to the LLM (propose/verify candidates).
    - Controller (controller.py): decides whether the LLM is needed
      (uncertainty gate) and whether to adopt its proposed fields
      (field-level evidence-gated adoption). No open-ended code-execution
      loop here; this is a single bounded verify-and-adopt step.
    """

    def __init__(self, agent_prompt, basic_prompt) -> None:
        self.ap = agent_prompt
        self.bp = basic_prompt

    def run(
        self,
        instruction,
        logger,
        dataset_name: str = "",
        task_focus: str = "",
        context: dict = None,
    ):
        logger.info(f"Objective: {instruction}")

        executor = CounterfactualExecutor(
            agent_prompt=self.ap,
            basic_prompt=self.bp,
            logger=logger,
        )
        controller = CounterfactualController(
            executor=executor,
            logger=logger,
        )

        prediction, trajectory, prompt = controller.run(
            instruction=instruction,
            dataset_name=dataset_name,
            task_focus=task_focus,
            context=context,
        )

        logger.info(f"Result: {prediction}")
        return prediction, trajectory, prompt
