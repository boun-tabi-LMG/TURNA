"""This script tracks whether a specific TPU VM is still running"""

import argparse
import subprocess
import discord


def check_tpu_preempted(tpu_name=None):
    """Checks whether a TPU is preempted.

        The output of the command is something like this:

        NAME       ZONE            ACCELERATOR_TYPE  TYPE  TOPOLOGY  NETWORK  RANGE          STATUS
    tpu-vm-05  europe-west4-a  v3-8              V3    2x2       default  10.164.0.0/20  PREEMPTED
    tpu-vm-04  europe-west4-a  v3-8              V3    2x2       default  10.164.0.0/20  PREEMPTED
    tpu-vm-08  europe-west4-a  v3-8              V3    2x2       default  10.164.0.0/20  READY

        Args:
            tpu_name (_type_): _description_

        Returns:
            _type_: _description_
    """
    command = (
        f"gcloud compute tpus tpu-vm list --filter='schedulingConfig.preemptible=true'"
    )

    output = subprocess.check_output(command, shell=True).decode("utf-8")

    tpu_dict = {}

    lines = output.strip().split("\n")
    for line in lines[1:]:
        columns = line.split()
        tpu_dict[columns[0]] = columns[-1]

    if tpu_name is None:
        tpu_names_statues = sorted(tpu_dict.items(), key=lambda x: x[0])
        return tpu_names_statues[-1][1] == "PREEMPTED", output, tpu_names_statues[-1][0]
    else:
        return (
            tpu_name in tpu_dict and tpu_dict[tpu_name] == "PREEMPTED",
            output,
            tpu_name,
        )


def send_message(message):
    token = "MTE3Njg5MDU4MzgyMjkwOTU2MQ.GfZDT8.ImNfCBSM5RLgObiJEJE1bJxQRXkuckCmRx-noM"

    # Discord channel ID
    channel_id = "1073904842956873758"

    intents = discord.Intents.default()
    intents.typing = False
    intents.presences = False

    # Create a Discord client
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        channel = client.get_channel(int(channel_id))
        await channel.send(message)
        await client.close()

    # Run the Discord client
    client.run(token)


parser = argparse.ArgumentParser()
parser.add_argument("--tpu_name", help="Name of the TPU", required=False, default=None)
parser.add_argument("--send_report", help="Name of the TPU", action="store_true")
args = parser.parse_args()

tpu_name = args.tpu_name

is_preempted, output, tpu_name = check_tpu_preempted(tpu_name)

if args.send_report:
    send_message(f"Report\n\n```bash\n{output}```")
elif is_preempted:
    # Discord bot token
    send_message(f"TPU {tpu_name} has been preempted!\n\n{output}")

else:
    print(f"TPU {tpu_name} is not preempted.")
