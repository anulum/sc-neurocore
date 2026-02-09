#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    let m = (1_i64 << width) - 1;
    let mut v = (value as i64) & m;
    if v >= (1_i64 << (width - 1)) {
        v -= 1_i64 << width;
    }
    if width >= 32 {
        // For extended intermediate masks (e.g. 2 * data_width), emulate
        // Python's signed interpretation before truncating to i16.
        v as i32 as i16
    } else {
        v as i16
    }
}

#[derive(Clone, Debug)]
pub struct FixedPointLif {
    pub v: i16,
    pub refractory_counter: i32,
    pub data_width: u32,
    pub fraction: u32,
    pub v_rest: i16,
    pub v_reset: i16,
    pub v_threshold: i16,
    pub refractory_period: i32,
}

impl FixedPointLif {
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self {
            v: v_rest,
            refractory_counter: 0,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        }
    }

    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let w = self.data_width;
        let diff = mask((self.v_rest as i32) - (self.v as i32), 2 * w) as i32;
        let leak_mul = diff * (leak_k as i32);
        let dv_leak = mask(leak_mul >> self.fraction, self.data_width);

        let in_mul = (i_t as i32) * (gain_k as i32);
        let dv_in = mask(in_mul >> self.fraction, self.data_width);

        let v_next = mask(
            (self.v as i32) + (dv_leak as i32) + (dv_in as i32) + (noise_in as i32),
            self.data_width,
        );

        let mut spike = if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_counter = self.refractory_period;
            1
        } else {
            self.v = v_next;
            0
        };

        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            spike = 0;
        }

        (spike, mask(self.v as i32, w))
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}
