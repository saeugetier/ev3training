use lqr::LQRController;
use nalgebra::{
    Matrix1, Matrix4, Matrix4x1, Vector4,
    U1, U4,
};

// Zustände:
//
// x[0] = Position       [m]
// x[1] = Geschwindigkeit [m/s]
// x[2] = Neigungswinkel [rad]
// x[3] = Winkelgeschwindigkeit [rad/s]
//
// Eingang:
// u = gemeinsames Motormoment
//
// Die Differenz der Motoren wird separat für die Drehung verwendet.

type BalanceLqr = LQRController<f64, U4, U1>;

pub struct BalanceBot {
    lqr: BalanceLqr,

    // Fahrwunsch
    target_velocity: f64,
    target_turn: f64,

    // letzte Zustände
    state: Vector4<f64>,
}

impl BalanceBot {
    pub fn new(dt: f64) -> Self {
        /*
         * Vereinfachtes diskretes Modell.
         *
         * x(k+1) = A*x(k) + B*u(k)
         *
         * Reihenfolge:
         * position
         * velocity
         * angle
         * angular velocity
         */

        let a = Matrix4::new(
            1.0, dt, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, dt,
            0.0, 0.0, 12.0 * dt, 1.0,
        );

        let b = Matrix4x1::new(
            0.0,
            1.5 * dt,
            0.0,
            -8.0 * dt,
        );

        /*
         * Q bestimmt, welche Zustände dem LQR besonders
         * wichtig sind.
         *
         * Hohe Gewichtung auf Winkel:
         * -> Roboter versucht aggressiver aufrecht zu bleiben.
         */

        let q = Matrix4::new(
            1.0, 0.0,   0.0,   0.0,
            0.0, 0.5,   0.0,   0.0,
            0.0, 0.0,  100.0,  0.0,
            0.0, 0.0,   0.0,  10.0,
        );

        // Kosten des Motoreinsatzes
        let r = Matrix1::new(0.1);

        let mut lqr = BalanceLqr::new().unwrap();

        // LQR-Verstärkung berechnen
        lqr.compute_gain(&a, &b, &q, &r, 1e-6 ).expect("LQR gain calculation failed");

        Self {
            lqr,
            target_velocity: 0.0,
            target_turn: 0.0,
            state: Vector4::zeros(),
        }
    }

    pub fn set_velocity(&mut self, velocity: f64) {
        self.target_velocity = velocity;
    }

    pub fn set_turn(&mut self, turn: f64) {
        self.target_turn = turn;
    }

    /*
     * Wird zyklisch aufgerufen, z.B. alle 5 ms.
     */
    pub fn update(
        &mut self,
        position: f64,
        velocity: f64,
        angle: f64,
        angular_velocity: f64,
    ) -> (f64, f64) {

        self.state = Vector4::new(
            position,
            velocity,
            angle,
            angular_velocity,
        );

        /*
         * Zielzustand.
         *
         * Für einen einfachen Balance-Bot:
         *
         * position = 0
         * velocity = gewünschte Geschwindigkeit
         * angle = 0
         * angular_velocity = 0
         */

        let target = Vector4::new(
            0.0,
            self.target_velocity,
            0.0,
            0.0,
        );

        /*
         * LQR berechnet das gemeinsame Antriebssignal.
         */
        let u_balance =
            self.lqr.compute_optimal_controls(
                &self.state,
                &target,
            ).unwrap();

        let drive = u_balance[0];

        /*
         * Differentialantrieb
         *
         * drive = Vorwärts/Rückwärts
         * turn  = Drehung
         */

        let left = drive - self.target_turn;
        let right = drive + self.target_turn;

        /*
         * Motoren begrenzen.
         */
        let left = left.clamp(-1.0, 1.0);
        let right = right.clamp(-1.0, 1.0);

        (left, right)
    }
}