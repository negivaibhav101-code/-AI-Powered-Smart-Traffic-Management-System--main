class MaxPressureController:
    def __init__(self, epoch=2.0, min_green=4.0, max_green=20.0, clearance=2.0, alpha=0.3,
                 spill_limit=14, emergency_at=0.0, emergency_for=0.0, emergency_dir="EW"):
        self.EPOCH = epoch
        self.MIN_GREEN = min_green
        self.MAX_GREEN = max_green
        self.CLEARANCE = clearance
        self.ALPHA = alpha
        self.SPILL_LIM = spill_limit
        self.emergency_at = emergency_at
        self.emergency_for = emergency_for
        self.emergency_dir = emergency_dir.upper()

        self.phase = 0
        self.t_in_phase = 0.0
        self.clear_left = 0.0
        self.since_decide = 0.0

    def _emg_active(self, t):
        if self.emergency_at <= 0 or self.emergency_for <= 0: return False
        return self.emergency_at <= t < (self.emergency_at + self.emergency_for)

    def _pressures(self, qN, qS, qE, qW, center_load):
        q_ns = qN + qS
        q_ew = qE + qW
        dn_ns = dn_ew = 0
        if center_load >= self.SPILL_LIM:
            dn_ns = center_load
            dn_ew = center_load
        pNS = q_ns - self.ALPHA*dn_ns
        pEW = q_ew - self.ALPHA*dn_ew
        return pNS, pEW

    def choose(self, dt, tnow, qN, qS, qE, qW, center_load):
        self.t_in_phase += dt
        self.since_decide += dt

        if self.clear_left > 0:
            self.clear_left -= dt
            return self.phase, True, self._emg_active(tnow)

        if self._emg_active(tnow):
            target = 1 if self.emergency_dir == "EW" else 0
            if self.phase != target and self.t_in_phase >= self.MIN_GREEN:
                self.clear_left = self.CLEARANCE
            return self.phase, False, True

        want_decide = self.since_decide >= self.EPOCH
        force_flip = self.t_in_phase >= self.MAX_GREEN

        if want_decide or force_flip:
            pNS, pEW = self._pressures(qN,qS,qE,qW,center_load)
            best = 0 if pNS >= pEW else 1

            if force_flip and self.clear_left <= 0:
                if self.t_in_phase >= self.MIN_GREEN:
                    self.clear_left = self.CLEARANCE
                    self.since_decide = 0.0
                    return self.phase, True, False

            if best != self.phase:
                if self.t_in_phase >= self.MIN_GREEN:
                    self.clear_left = self.CLEARANCE
                    self.since_decide = 0.0
                    return self.phase, True, False

            self.since_decide = 0.0

        return self.phase, False, False

    def apply_after_clearance(self):
        if self.clear_left <= 0:
            self.phase = 1 - self.phase
            self.t_in_phase = 0.0