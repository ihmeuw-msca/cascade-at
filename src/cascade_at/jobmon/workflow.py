import os
import getpass

from cascade_at.core.db import swarm
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)

Workflow = swarm.workflow.workflow.Workflow
BashTask = swarm.workflow.bash_task.BashTask
ExecutorParameters = swarm.executors.base.ExecutorParameters


class COBashTask(BashTask):
    """
    Just a little modification to BashTask so that it has
    an attribute for upstream commands in order for us to index
    the cascade operations correctly.
    """
    def __init__(self, upstream_commands=None, **kwargs):
        super().__init__(**kwargs)
        if upstream_commands is None:
            upstream_commands = []
        self.upstream_commands = upstream_commands
        LOG.info(f"Created task with command {self.command} with "
                 f"{len(self.upstream_commands)} upstream commands.")


def bash_task_from_cascade_operation(co):
    """
    Create a bash task from a cascade operation (co for short)

    :param co: (cascade_at.cascade.cascade_operations.CascadeOperation)
    :return: jobmon.client.swarm.workflow.bash_task.BashTask
    """
    name = co.command.split(' ')
    return COBashTask(
        command=co.command,
        name='_'.join([name[0], name[2]]),
        upstream_commands=co.upstream_commands,
        executor_parameters=ExecutorParameters(
            max_runtime_seconds=co.executor_parameters['max_runtime_seconds'],
            j_resource=co.j_resource,
            m_mem_free=co.executor_parameters['m_mem_free'],
            num_cores=co.executor_parameters['num_cores'],
            resource_scales=co.executor_parameters['resource_scales']
        )
    )


def jobmon_workflow_from_cascade_command(cc, context):
    """
    Create a jobmon workflow from a cascade command (cc for short)

    :param cc: (cascade_at.cascade.cascade_commands.CascadeCommand)
    :param context: (cascade_at.context.model_context.Context)
    :return: jobmon.client.swarm.workflow.workflow.Workflow
    """
    user = getpass.getuser()

    error_dir = context.log_dir / 'errors'
    output_dir = context.log_dir / 'output'

    for folder in [error_dir, output_dir]:
        os.makedirs(folder, exist_ok=True)

    wf = Workflow(
        workflow_args=f'dismod-at_{cc.model_version_id}',
        project='proj_dismod_at',
        stderr=str(error_dir),
        stdout=str(output_dir),
        working_dir=str(context.model_dir),
        seconds_until_timeout=60*60*24*5,
        resume=True
    )
    bash_tasks = {command: bash_task_from_cascade_operation(co)
                  for command, co in cc.task_dict.items()}
    for command, task in bash_tasks.items():
        for upstream in task.upstream_commands:
            task.add_upstream(bash_tasks.get(upstream))

    wf.add_tasks(list(bash_tasks.values()))
    return wf

