/**
 * Supabase Client Initialization (TypeScript)
 * Supports Vite, Next.js, and standard ES Module environments.
 */

import { createClient } from '@supabase/supabase-js';

const getEnvVar = (key: string, fallback: string): string => {
  // Check Vite import.meta.env
  if (typeof import.meta !== 'undefined' && (import.meta as any).env && (import.meta as any).env[key]) {
    return (import.meta as any).env[key];
  }
  // Check Node / Next.js / CRA process.env
  if (typeof process !== 'undefined' && process.env && process.env[key]) {
    return process.env[key] as string;
  }
  return fallback;
};

const supabaseUrl = getEnvVar('VITE_SUPABASE_URL', getEnvVar('NEXT_PUBLIC_SUPABASE_URL', 'https://dkmxjeuoidbxnpwrjadv.supabase.co'));
const supabaseAnonKey = getEnvVar('VITE_SUPABASE_ANON_KEY', getEnvVar('NEXT_PUBLIC_SUPABASE_ANON_KEY', 'sb_publishable_Y_MAxfihttak5yXTLta95w_0QgL7'));

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('[Supabase] Missing Supabase URL or Anon Key. Using project defaults.');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
export default supabase;
