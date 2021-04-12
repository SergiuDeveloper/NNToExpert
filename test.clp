(defrule r0
	(input_vars ?V0 ?V1)
	(test (<= ?V1 0.5))
=>
	(assert (output_vars 0))
)

(defrule r1
	(input_vars ?V0 ?V1)
	(test (> ?V1 0.5))
	(test (<= ?V0 0.5))
=>
	(assert (output_vars 0))
)

(defrule r2
	(input_vars ?V0 ?V1)
	(test (> ?V1 0.5))
	(test (> ?V0 0.5))
=>
	(assert (output_vars 1))
)
