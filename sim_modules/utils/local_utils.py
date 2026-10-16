


class LinearStepAnneal(object):
    # def __init__(self, total_iters, start_state=[0.02, 0.98], end_state=[0.50, 0.98]):
    def __init__(
        self,
        total_iters,
        start_state=[0.02, 0.98],
        end_state=[0.02, 0.98],
        plateau_iters=-1,
        warmup_step=300,
    ):
        self.total_iters = total_iters

        if plateau_iters < 0:
            plateau_iters = int(total_iters * 0.2)

        if warmup_step <= 0:
            warmup_step = 0

        self.total_iters = max(total_iters - plateau_iters - warmup_step, 10)

        self.start_state = start_state
        self.end_state = end_state
        self.warmup_step = warmup_step

    def compute_state(self, cur_iter):
        """
        根据当前迭代次数 cur_iter，计算一个线性插值后的状态（state）值
        
        我估计window_step会逐渐变化，随着迭代轮次
        """
        if self.warmup_step > 0:
            cur_iter = max(0, cur_iter - self.warmup_step)
        if cur_iter >= self.total_iters:
            return self.end_state
        ret = []
        for s, e in zip(self.start_state, self.end_state):
            ret.append(s + (e - s) * cur_iter / self.total_iters)
        return ret