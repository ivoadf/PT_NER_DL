import torch
from torch.autograd import Variable

############################################################################
# Functions to compute in log-domain.
############################################################################


class LogOps():
    def __init__(self, gpu):
        if gpu and torch.cuda.is_available():
            self.float_type = torch.cuda.FloatTensor
        else:
            self.float_type = torch.FloatTensor
        self.inf_tensor = Variable(self.float_type([-float('inf')]))

    def logzero(self):
        return self.inf_tensor

    def safe_log(self, x):
        if x == 0:
            return self.logzero()
        return torch.log(x)

    def logsum_pair(self, logx, logy):
        '''
        Return log(x+y), avoiding arithmetic underflow/overflow.

        logx: log(x)
        logy: log(y)

        Rationale:

        x + y    = e^logx + e^logy
                 = e^logx (1 + e^(logy-logx))
        log(x+y) = logx + log(1 + e^(logy-logx)) (1)

        Likewise,
        log(x+y) = logy + log(1 + e^(logx-logy)) (2)

        The computation of the exponential overflows earlier and is less precise
        for big values than for small values. Due to the presence of logy-logx
        (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
        otherwise.
        '''
        if torch.equal(logx, self.logzero()):
            return logy
        elif torch.gt(logx, logy).any():
            return logx + torch.log1p(torch.exp(logy-logx))
        else:
            return logy + torch.log1p(torch.exp(logx-logy))

    def logsum(self, logv):
        '''
        Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
        '''
        res = self.logzero()
        for val in logv:
            res = self.logsum_pair(res, val)
        return res

    def logsumexp(self, inputs):
        dim = 0
        s, _ = torch.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        outputs = outputs.squeeze(dim)
        return outputs


############################################################################
# This implementation is faster, but may give problems with log(0), so I
# commented it out
# def logsum(logv):
#     '''
#     Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
#     '''
#     c = np.max(logv)
#     return c + np.log(np.sum(np.exp(logv - c)))
############################################################################
