CREATE TABLE `downloads` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL,
	`fmt` text NOT NULL,
	`row_count` integer NOT NULL,
	`filters` text,
	`email` text,
	`name` text,
	`ip` text,
	`user_agent` text,
	`referer` text
);
--> statement-breakpoint
CREATE INDEX `idx_dl_created_at` ON `downloads` (`created_at`);--> statement-breakpoint
CREATE INDEX `idx_dl_email` ON `downloads` (`email`);--> statement-breakpoint
CREATE TABLE `mutation_class_payloads` (
	`mutation_class_id` text PRIMARY KEY NOT NULL,
	`labeled_quivers` text NOT NULL,
	`boundary_quivers` text NOT NULL,
	FOREIGN KEY (`mutation_class_id`) REFERENCES `mutation_classes`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `mutation_classes` (
	`id` text PRIMARY KEY NOT NULL,
	`n` integer NOT NULL,
	`canonical_matrix` text NOT NULL,
	`canonical_quiver_id` text,
	`is_open` integer NOT NULL,
	`class_size` integer NOT NULL,
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
);
--> statement-breakpoint
CREATE INDEX `idx_mc_n_id` ON `mutation_classes` (`n`,`id`);--> statement-breakpoint
CREATE INDEX `idx_mc_is_open` ON `mutation_classes` (`is_open`);--> statement-breakpoint
CREATE INDEX `idx_mc_dynkin_type` ON `mutation_classes` (`dynkin_type`);--> statement-breakpoint
CREATE INDEX `idx_mc_class_size` ON `mutation_classes` (`class_size`);--> statement-breakpoint
CREATE INDEX `idx_mc_is_mutation_acyclic` ON `mutation_classes` (`is_mutation_acyclic`);--> statement-breakpoint
CREATE INDEX `idx_mc_is_banff` ON `mutation_classes` (`is_banff`);--> statement-breakpoint
CREATE INDEX `idx_mc_is_louise` ON `mutation_classes` (`is_louise`);--> statement-breakpoint
CREATE INDEX `idx_mc_is_p_prime` ON `mutation_classes` (`is_p_prime`);--> statement-breakpoint
CREATE TABLE `quivers` (
	`id` text PRIMARY KEY NOT NULL,
	`n` integer NOT NULL,
	`exchange_matrix` text NOT NULL,
	`mutation_class_id` text,
	`max_edge` integer DEFAULT 0 NOT NULL,
	`is_acyclic` integer DEFAULT true NOT NULL,
	`is_connected` integer DEFAULT true NOT NULL,
	`is_bipartite` integer,
	`is_abundant` integer,
	`is_planar` integer,
	`labeling_count` integer,
	`representation_type` text,
	`symmetry_group` text,
	FOREIGN KEY (`mutation_class_id`) REFERENCES `mutation_classes`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `idx_q_n_id` ON `quivers` (`n`,`id`);--> statement-breakpoint
CREATE INDEX `idx_q_mutation_class_id` ON `quivers` (`mutation_class_id`);--> statement-breakpoint
CREATE INDEX `idx_q_max_edge` ON `quivers` (`max_edge`);--> statement-breakpoint
CREATE INDEX `idx_q_is_acyclic` ON `quivers` (`is_acyclic`);--> statement-breakpoint
CREATE INDEX `idx_q_is_connected` ON `quivers` (`is_connected`);--> statement-breakpoint
CREATE INDEX `idx_q_is_bipartite` ON `quivers` (`is_bipartite`);--> statement-breakpoint
CREATE INDEX `idx_q_representation_type` ON `quivers` (`representation_type`);--> statement-breakpoint
CREATE TABLE `rank_stats` (
	`n` integer PRIMARY KEY NOT NULL,
	`quiver_count` integer NOT NULL,
	`labeled_quiver_count` integer NOT NULL,
	`class_count` integer NOT NULL
);
