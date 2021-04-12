(defrule MAIN::r0
   (input_vars ?V0 ?V1)
   (test (<= ?V1 0.5))
   (test (<= ?V0 0.5))
   =>
   (assert (output_vars 0)))

(defrule MAIN::r1
   (input_vars ?V0 ?V1)
   (test (<= ?V1 0.5))
   (test (> ?V0 0.5))
   =>
   (assert (output_vars 1)))

(defrule MAIN::r2
   (input_vars ?V0 ?V1)
   (test (> ?V1 0.5))
   (test (<= ?V0 0.5))
   =>
   (assert (output_vars 1)))

(defrule MAIN::r3
   (input_vars ?V0 ?V1)
   (test (> ?V1 0.5))
   (test (> ?V0 0.5))
   =>
   (assert (output_vars 0)))

