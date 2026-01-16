import os
from agentset import Agentset
import dotenv
dotenv.load_dotenv()

client = Agentset(
    namespace_id="ns_cmkgncf8s000104l5l1i4rfq7",
    token=os.environ["AGENTSET_API_KEY"],
)

job = client.ingest_jobs.create(
    name="Attention Is All You Need",
    payload={
        "type": "FILE",
        "fileUrl": "https://arxiv.org/pdf/1706.03762.pdf",
    }
)

print(f"Uploaded: {job.data.id}")