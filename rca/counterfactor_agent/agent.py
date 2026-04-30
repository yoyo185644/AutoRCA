from .controller import CounterfactualController
from .executor import CounterfactualExecutor


class CounterfactualRCAAgent:
    def __init__(self, agent_prompt, basic_prompt) -> None:
        self.ap = agent_prompt
        self.bp = basic_prompt

    def run(
        self,
        instruction,
        logger,
        dataset_name: str = "",
        max_step: int = 25,
        max_turn: int = 5,
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
            max_step=max_step,
            max_turn=max_turn,
            task_focus=task_focus,
            context=context,
        )

        logger.info(f"Result: {prediction}")
        return prediction, trajectory, prompt