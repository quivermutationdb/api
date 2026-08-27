CREATE TABLE `class_nicknames` (
	`mc_id` text PRIMARY KEY NOT NULL,
	`nickname` text NOT NULL,
	`slug` text NOT NULL,
	`note` text,
	`added_by` text,
	`added_at` text
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_nick_slug` ON `class_nicknames` (`slug`);--> statement-breakpoint
CREATE TABLE `frontier_quivers` (
	`mutation_class_id` text NOT NULL,
	`ord` integer NOT NULL,
	`matrix` text NOT NULL,
	PRIMARY KEY(`mutation_class_id`, `ord`),
	FOREIGN KEY (`mutation_class_id`) REFERENCES `mutation_classes`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `labelings` (
	`mutation_class_id` text NOT NULL,
	`ord` integer NOT NULL,
	`qmd_id` text NOT NULL,
	`matrix` text NOT NULL,
	PRIMARY KEY(`mutation_class_id`, `ord`),
	FOREIGN KEY (`mutation_class_id`) REFERENCES `mutation_classes`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `idx_lab_qmd_ord` ON `labelings` (`qmd_id`,`ord`);--> statement-breakpoint
CREATE INDEX `idx_lab_mc_qmd_ord` ON `labelings` (`mutation_class_id`,`qmd_id`,`ord`);--> statement-breakpoint
DROP TABLE `mutation_class_payloads`;--> statement-breakpoint
DROP INDEX `idx_mc_is_open`;--> statement-breakpoint
DROP INDEX `idx_mc_dynkin_type`;--> statement-breakpoint
DROP INDEX `idx_mc_class_size`;--> statement-breakpoint
DROP INDEX `idx_mc_is_banff`;--> statement-breakpoint
DROP INDEX `idx_mc_is_louise`;--> statement-breakpoint
DROP INDEX `idx_mc_is_p_prime`;--> statement-breakpoint
ALTER TABLE `mutation_classes` ADD `exploration` text DEFAULT 'complete' NOT NULL;--> statement-breakpoint
CREATE INDEX `idx_mc_n_class_size_id` ON `mutation_classes` (`n`,`class_size`,`id`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_dynkin_id` ON `mutation_classes` (`n`,`dynkin_type`,`id`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_distinct_id` ON `mutation_classes` (`n`,`distinct_quiver_count`,`id`);--> statement-breakpoint
CREATE INDEX `idx_mc_n_open_id` ON `mutation_classes` (`n`,`is_open`,`id`);--> statement-breakpoint
CREATE INDEX `idx_mc_finite_confirmed` ON `mutation_classes` (`is_finite_confirmed`);--> statement-breakpoint
CREATE INDEX `idx_mc_infinite_confirmed` ON `mutation_classes` (`is_infinite_confirmed`);--> statement-breakpoint
DROP INDEX `idx_q_mutation_class_id`;--> statement-breakpoint
DROP INDEX `idx_q_max_edge`;--> statement-breakpoint
DROP INDEX `idx_q_is_acyclic`;--> statement-breakpoint
DROP INDEX `idx_q_is_connected`;--> statement-breakpoint
DROP INDEX `idx_q_is_bipartite`;--> statement-breakpoint
ALTER TABLE `quivers` ADD `labeling_offset` integer;--> statement-breakpoint
CREATE INDEX `idx_q_n_max_edge_id` ON `quivers` (`n`,`max_edge`,`id`);--> statement-breakpoint
CREATE INDEX `idx_q_n_labeling_offset` ON `quivers` (`n`,`labeling_offset`);--> statement-breakpoint
CREATE INDEX `idx_q_mc_labcount_id` ON `quivers` (`mutation_class_id`,`labeling_count`,`id`);--> statement-breakpoint
ALTER TABLE `rank_stats` ADD `bound` integer;--> statement-breakpoint
ALTER TABLE `rank_stats` ADD `node_cap` integer;--> statement-breakpoint
ALTER TABLE `rank_stats` ADD `generated_at` text;--> statement-breakpoint
ALTER TABLE `rank_stats` ADD `pipeline_version` text;