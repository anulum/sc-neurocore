#[derive(Clone, Debug)]
pub struct Lfsr16 {
    pub reg: u16,
    pub width: u32,
}

impl Lfsr16 {
    pub fn new(seed: u16) -> Self {
        assert_ne!(seed, 0, "LFSR seed must be non-zero.");
        Self {
            reg: seed,
            width: 16,
        }
    }

    pub fn step(&mut self) -> u16 {
        let feedback = ((self.reg >> 15) ^ (self.reg >> 13) ^ (self.reg >> 12) ^ (self.reg >> 10)) & 1;
        self.reg = ((self.reg << 1) & 0xffff) | feedback;
        self.reg
    }

    pub fn reset(&mut self, seed: Option<u16>) {
        let next = seed.unwrap_or(self.reg);
        assert_ne!(next, 0, "LFSR seed must be non-zero.");
        self.reg = next;
    }
}

#[derive(Clone, Debug)]
pub struct BitstreamEncoder {
    pub lfsr: Lfsr16,
    pub data_width: u32,
    seed_init: u16,
}

impl BitstreamEncoder {
    pub fn new(data_width: u32, seed: u16) -> Self {
        Self {
            lfsr: Lfsr16::new(seed),
            data_width,
            seed_init: seed,
        }
    }

    pub fn step(&mut self, x_value: u16) -> u8 {
        self.lfsr.step();
        if self.lfsr.reg < x_value {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self, seed: Option<u16>) {
        let next = seed.unwrap_or(self.seed_init);
        self.lfsr = Lfsr16::new(next);
    }
}
