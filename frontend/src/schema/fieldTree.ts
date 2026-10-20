import { controllerOf } from "../config/visibility";
import type { Field } from "./groups";

/** A field and the shown fields that depend on it (visibility.ts), drawn as one block. */
export interface FieldNode {
  field: Field;
  children: FieldNode[];
}

// rules on these pick fields per page; they are not options the user switches on
const NOT_A_PARENT = new Set(["model_train_type"]);

/** Puts each field under its controller when that is shown in the same list, keeping the list's order
 * among siblings; a field whose controller is elsewhere (another section, hidden by a search or by
 * Essentials) stays at its own place. */
export function fieldTree(fields: Field[]): FieldNode[] {
  const nodes = new Map(fields.map((field) => [field.name, { field, children: [] } as FieldNode]));
  const roots: FieldNode[] = [];
  for (const field of fields) {
    const node = nodes.get(field.name)!;
    const controller = controllerOf(field.name);
    const parent = controller && !NOT_A_PARENT.has(controller) ? nodes.get(controller) : undefined;
    (parent ? parent.children : roots).push(node);
  }
  return roots;
}
