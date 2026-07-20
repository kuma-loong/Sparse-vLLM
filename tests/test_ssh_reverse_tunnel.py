import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmark" / "ssh_reverse_tunnel.sh"


class ReverseSshTunnelTest(unittest.TestCase):
    def test_script_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_failed_ssh_is_retried_with_a_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_ssh = root / "fake-ssh"
            calls = root / "calls"
            fake_ssh.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$FAKE_SSH_CALLS\"\nexit 23\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(fake_ssh.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env.update(
                {
                    "SSH_BIN": str(fake_ssh),
                    "SSH_DESTINATION": "user@example.test",
                    "SSH_REVERSE_FORWARD": "127.0.0.1:18000:127.0.0.1:18000",
                    "SSH_JUMP_HOST": "jump@example.test",
                    "SSH_MAX_RECONNECTS": "2",
                    "SSH_RECONNECT_DELAY_S": "0",
                    "SSH_LOG": str(root / "ssh.log"),
                    "SSH_STATUS_FILE": str(root / "status.tsv"),
                    "FAKE_SSH_CALLS": str(calls),
                }
            )

            result = subprocess.run(["bash", str(SCRIPT)], env=env)

            self.assertEqual(result.returncode, 23)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 3)
            states = (root / "status.tsv").read_text(encoding="utf-8")
            self.assertEqual(states.count("\tconnecting\t"), 3)
            self.assertIn("\tfailed\t3\t23", states)


if __name__ == "__main__":
    unittest.main()
