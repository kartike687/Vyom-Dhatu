/**
 * Supabase Client Initialization (JavaScript)
 * Exports initialized Supabase client using @supabase/supabase-js.
 */

import { createClient } from '@supabase/supabase-js';

const getEnvVar = (key, fallback) => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env[key]) {
    return import.meta.env[key];
  }
  if (typeof process !== 'undefined' && process.env && process.env[key]) {
    return process.env[key];
  }
  return fallback;
};

const supabaseUrl = getEnvVar('VITE_SUPABASE_URL', getEnvVar('NEXT_PUBLIC_SUPABASE_URL', 'https://dkmxjeuoidbxnpwrjadv.supabase.co'));
const supabaseAnonKey = getEnvVar('VITE_SUPABASE_ANON_KEY', getEnvVar('NEXT_PUBLIC_SUPABASE_ANON_KEY', 'sb_publishable_Y_MAxfihttak5yXTLta95w_0QgL7'));

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
export default supabase;
