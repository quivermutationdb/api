-- Schema v3 (docs/PHASE3.md): compact matrices, rowid-based indexes,
-- per-quiver mutation_finite, labelings only for complete classes, no frontier.
-- Every rank is re-imported after this migration, so the row tables are
-- recreated rather than migrated in place.
DROP TABLE IF EXISTS `frontier_quivers`;--> statement-breakpoint
DROP TABLE IF EXISTS `labelings`;--> statement-breakpoint
DROP TABLE IF EXISTS `quivers`;--> statement-breakpoint
DROP TABLE IF EXISTS `mutation_classes`;--> statement-breakpoint
CREATE TABLE `mutation_classes` (
	`id` text PRIMARY KEY NOT NULL,
	`n` integer NOT NULL,
	`canonical_matrix` text NOT NULL,
	`canonical_quiver_id` text,
	`is_open` integer NOT NULL,
	`exploration` text DEFAULT 'complete' NOT NULL,
	`class_size` integer,
	`distinct_quiver_count` integer NOT NULL,
	`merged_orbit_count` integer DEFAULT 1 NOT NULL,
	`dynkin_type` text,
	`label` text,
	`is_finite_confirmed` integer,
	`is_infinite_confirmed` integer,
	`is_infinite_expected` integer,
	`size_of_explored_frontier` integer,
	`is_mutation_acyclic` integer,
	`is_banff` integer,
	`is_louise` integer,
	`is_p_prime` integer,
	`provenance` text
);--> statement-breakpoint
CREATE INDEX `idx_mc_n` ON `mutation_classes` (`n`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_class_size` ON `mutation_classes` (`n`,`class_size`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_dynkin` ON `mutation_classes` (`n`,`dynkin_type`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_distinct` ON `mutation_classes` (`n`,`distinct_quiver_count`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_open` ON `mutation_classes` (`n`,`is_open`);--> statement-breakpoint
CREATE INDEX `idx_mc_finite_confirmed` ON `mutation_classes` (`is_finite_confirmed`);--> statement-breakpoint
CREATE INDEX `idx_mc_infinite_confirmed` ON `mutation_classes` (`is_infinite_confirmed`);--> statement-breakpoint
CREATE INDEX `idx_mc_is_mutation_acyclic` ON `mutation_classes` (`is_mutation_acyclic`);--> statement-breakpoint
CREATE TABLE `labelings` (
	`mutation_class_id` text NOT NULL,
	`ord` integer NOT NULL,
	`qmd_id` text NOT NULL,
	`matrix` text NOT NULL,
	PRIMARY KEY(`mutation_class_id`, `ord`),
	FOREIGN KEY (`mutation_class_id`) REFERENCES `mutation_classes`(`id`) ON UPDATE no action ON DELETE cascade
);--> statement-breakpoint
CREATE INDEX `idx_lab_qmd_ord` ON `labelings` (`qmd_id`,`ord`);--> statement-breakpoint
CREATE TABLE `quivers` (
	`id` text PRIMARY KEY NOT NULL,
	`n` integer NOT NULL,
	`exchange_matrix` text NOT NULL,
	`mutation_class_id` text,
	`mutation_finite` integer,
	`max_edge` integer DEFAULT 0 NOT NULL,
	`is_acyclic` integer DEFAULT true NOT NULL,
	`is_connected` integer DEFAULT true NOT NULL,
	`is_bipartite` integer,
	`is_abundant` integer,
	`is_planar` integer,
	`labeling_count` integer,
	`representation_type` text,
	`symmetry_group` text
);--> statement-breakpoint
CREATE INDEX `idx_q_n` ON `quivers` (`n`);--> statement-breakpoint
CREATE INDEX `idx_q_n_max_edge` ON `quivers` (`n`,`max_edge`);--> statement-breakpoint
CREATE INDEX `idx_q_n_finite` ON `quivers` (`n`,`mutation_finite`);--> statement-breakpoint
CREATE INDEX `idx_q_representation_type` ON `quivers` (`representation_type`);--> statement-breakpoint
CREATE INDEX `idx_q_mc_labcount` ON `quivers` (`mutation_class_id`,`labeling_count`);
--> statement-breakpoint
ALTER TABLE `rank_stats` ADD `shard_counts` text;
