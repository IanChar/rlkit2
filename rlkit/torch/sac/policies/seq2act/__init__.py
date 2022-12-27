from rlkit.torch.sac.policies.seq2act.sd_policy import SDPolicy, SDPolicyAdapter
from rlkit.torch.sac.policies.seq2act.sid_policy import SIDPolicy, SIDPolicyAdapter
from rlkit.torch.sac.policies.seq2act.sl_policy import SLPolicy, SLPolicyAdapter
from rlkit.torch.sac.policies.seq2act.frame_stacked_policy import (
    FrameStackPolicy,
    FrameStackPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.gaussian_seq import (
    SeqGaussianPolicy,
    SeqGaussianPolicyAdapter,
)

ADAPTER_DICT = {
    'sd': SDPolicyAdapter,
    'sid': SIDPolicyAdapter,
    'sl': SLPolicyAdapter,
    'frame_stack': FrameStackPolicyAdapter,
    'gaussian': SeqGaussianPolicyAdapter,
}
