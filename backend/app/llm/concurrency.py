"""Join model work before publishing a stage's terminal state."""

import asyncio


async def gather_complete(*awaitables):
    tasks = [asyncio.ensure_future(value) for value in awaitables]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results
